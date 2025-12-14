#!/usr/bin/env python
"""
Run with python train.py --config <config name>
"""

import argparse
import os
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm


def main():
    parser = argparse.ArgumentParser(description="Train E2C")
    parser.add_argument(
        "--config",
        dest="config_module",
        type=str,
        help="Config module name or dotted path (e.g., 'base_config' or 'configs.exp1')",
    )
    parser.add_argument("--seed", type=int, help="Random seed override per run")
    parser.add_argument(
        "--vae-type",
        choices=["conv", "mlp"],
        help="Override VAE type for this run (defaults to config.vae_type)",
    )
    parser.add_argument(
        "--dynamics",
        choices=["mlp", "rnn"],
        help="Override dynamics model type for this run (defaults to config.latent_dynamics_model_type)",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Enable torch.compile. If compilation fails, we fall back to eager mode.",
    )
    args = parser.parse_args()

    config_module = os.environ.get("E2C_CONFIG", "configs.base_config")

    if args.config_module:
        module = args.config_module
        if not module.startswith("configs."):
            module = f"configs.{module}"
        os.environ["E2C_CONFIG"] = module
        config_module = module
    elif not config_module.startswith("configs."):
        config_module = f"configs.{config_module}"
        os.environ["E2C_CONFIG"] = config_module

    from config import e2c_config as config
    from mujoco_data import create_mujoco_dataloader
    from models.e2c.rnn_e2c import RecurrentE2C
    from models.e2c.mlp_e2c import MLP_E2C
    from utils import get_device, set_random_seed, get_data_shape
    from engine import train_epoch, validate
    from run_utils import create_run_dir, save_config_json, get_checkpoint_path

    if args.seed is not None:
        config.random_seed = args.seed
    if args.vae_type:
        config.vae_type = args.vae_type
    if args.dynamics:
        config.latent_dynamics_model_type = args.dynamics

    set_random_seed(seed=config.random_seed)
    device = get_device()

    experiment_name = config_module.split(".")[-1]
    seed_value = config.random_seed
    x_dim, u_dim, T = get_data_shape()
    side = int(np.sqrt(x_dim))
    assert side * side == x_dim
    
    train_loader = create_mujoco_dataloader(split="train")
    val_loader = create_mujoco_dataloader(split="val")
    
    dynamics_type = config.latent_dynamics_model_type
    model_tag = f"{dynamics_type}-{config.vae_type}"
    run_dir = create_run_dir(
        model_tag,
        base_dir="checkpoints",
        experiment_name=experiment_name,
        seed=seed_value,
    )
    save_config_json(config, run_dir)
    
    print(f"Training {model_tag.upper()} E2C")
    print(f"Run directory: {run_dir}\n")
    
    if dynamics_type == "rnn":
        model = RecurrentE2C(x_dim=x_dim, u_dim=u_dim).to(device)
    elif dynamics_type == "mlp":
        model = MLP_E2C(x_dim=x_dim, u_dim=u_dim).to(device)
    else:
        raise ValueError(f"Unknown model_type: {dynamics_type}. Must be 'rnn' or 'mlp'")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, betas=(0.9, 0.999))
    
    # torch.compile brrrrr
    compiled_model = model
    if args.compile:
        try:
            compiled_model = torch.compile(model)
            print("[INFO] torch.compile enabled.")
        except Exception as e:
            compiled_model = model
            print(f"[WARN] torch.compile failed, using eager mode: {e}")

    writer = SummaryWriter(log_dir=str(run_dir / "logs"))
    
    global_step = 0
    
    for epoch in tqdm(range(config.epochs), desc="Epochs"):
        train_loss, train_kl, train_l_x, train_L_x_tp1, train_mse_x, train_mse_x_tp1, global_step = train_epoch(
            compiled_model, train_loader, optimizer, device, writer, global_step
        )
        
        val_loss, val_kl, val_l_x, val_L_x_tp1, val_mse_x, val_mse_x_tp1 = validate(compiled_model, val_loader, device)
        
        print(
            f"epoch {epoch+1:2d}/{config.epochs} | "
            f"train_loss: {train_loss:.4f} val_loss: {val_loss:.4f} | "
            f"train_kl: {train_kl:.3f} val_kl: {val_kl:.3f} | "
            f"train_mse: {train_mse_x:.4f}/{train_mse_x_tp1:.4f} "
            f"val_mse: {val_mse_x:.4f}/{val_mse_x_tp1:.4f}"
        )
        
        writer.add_scalars("loss/epoch", {
            "train": train_loss,
            "val": val_loss,
        }, epoch)
        
        writer.add_scalars("kl/epoch", {
            "train": train_kl,
            "val": val_kl,
        }, epoch)
        
        if (epoch + 1) % config.epochs_per_checkpoint == 0:
            checkpoint_path = get_checkpoint_path(run_dir, epoch + 1)
            torch.save({
                "epoch": epoch,
                "global_step": global_step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

    writer.close()
    print(f"Training complete, checkpoints saved to: {run_dir}")


if __name__ == "__main__":
    main()