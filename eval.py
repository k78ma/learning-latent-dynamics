#!/usr/bin/env python

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import imageio
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch
from tqdm import tqdm

from config import e2c_config
from mujoco_data import create_mujoco_dataloader
from models.e2c.common import kl_diag_gaussian, recons_loss, mse_loss
from models.e2c.mlp_e2c import MLP_E2C
from models.e2c.rnn_e2c import RecurrentE2C
from run_utils import create_run_dir, get_latest_checkpoint, parse_seed_dir, seed_dir_name
from utils import get_data_shape, get_device


def parse_model_tag(tag: str) -> Tuple[str, str]:
    if "-" in tag:
        dynamics_type, vae_type = tag.split("-", 1)
    else:
        dynamics_type, vae_type = tag, e2c_config.vae_type
    return dynamics_type, vae_type


def format_experiment_label(experiment_name: str) -> str:
    if experiment_name == "exp_baseline":
        return "Baseline Experiment"
    m = re.match(r"exp(\\d+)$", experiment_name)
    if m:
        return f"Experiment {m.group(1)}"
    return experiment_name


def format_model_label(model_tag: str) -> Tuple[str, str]:
    dyn, vae = parse_model_tag(model_tag)
    vae_label = "Convolutional VAE" if vae == "conv" else f"{vae.upper()} VAE"
    transition_label = "MLP Transition" if dyn == "mlp" else ("RNN Transition" if dyn == "rnn" else dyn.upper())
    return vae_label, transition_label


def format_plot_title(experiment_name: str, model_tag: str, runs: Optional[int] = None, epoch: Optional[int] = None) -> str:
    exp_label = format_experiment_label(experiment_name)
    vae_label, transition_label = format_model_label(model_tag)
    parts = [exp_label, vae_label, transition_label]
    if runs is not None:
        parts.append(f"{runs} run" + ("s" if runs != 1 else ""))
    if epoch is not None:
        parts.append(f"epoch {epoch}")
    return ", ".join(parts)


def extract_run_metadata(checkpoint_path: Path) -> Tuple[str, Optional[int], str, str]:
    resolved = checkpoint_path.resolve()
    experiment_name = "default_experiment"
    seed_value: Optional[int] = None
    model_tag = resolved.parent.parent.name
    timestamp_dir = resolved.parent.name

    parts = resolved.parts
    if "checkpoints" in parts:
        idx = parts.index("checkpoints")
        after = parts[idx + 1 :]
        if len(after) >= 1:
            experiment_name = after[0]
        if len(after) >= 2:
            seed_value = parse_seed_dir(after[1])
        if len(after) >= 3:
            model_tag = after[2]
        if len(after) >= 4:
            timestamp_dir = after[3]

    return experiment_name, seed_value, model_tag, timestamp_dir


def load_model_from_checkpoint(checkpoint_path: str, device: str):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    checkpoint_path = Path(checkpoint_path)
    _, _, model_tag, _ = extract_run_metadata(checkpoint_path)
    dynamics_type, vae_type = parse_model_tag(model_tag)

    e2c_config.latent_dynamics_model_type = dynamics_type
    e2c_config.vae_type = vae_type

    x_dim, u_dim, _ = get_data_shape()

    if dynamics_type == "rnn":
        model = RecurrentE2C(x_dim=x_dim, u_dim=u_dim)
    elif dynamics_type == "mlp":
        model = MLP_E2C(x_dim=x_dim, u_dim=u_dim)
    else:
        raise ValueError(f"Unknown dynamics type: {dynamics_type}")

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, model_tag, checkpoint


def evaluate_rollout(model, test_loader, device) -> dict:
    all_L_x = []
    all_L_x_tp1 = []
    all_KL = []
    all_MSE_x = []
    all_MSE_x_tp1 = []

    with torch.no_grad():
        for x_seq, u_seq in tqdm(test_loader, desc="Evaluating"):
            x_seq = x_seq.to(device)  # (B, K+1, C, H, W)
            u_seq = u_seq.to(device)  # (B, K, u_dim)

            B, Kp1, _, _, _ = x_seq.size()
            K = Kp1 - 1

            seq_L_x = []
            seq_L_x_tp1 = []
            seq_KL = []
            seq_MSE_x = []
            seq_MSE_x_tp1 = []

            # For RNN: maintain hidden state
            if hasattr(model.transition, "init_state"):
                h_t = model.transition.init_state(B, device)

            for t in range(K):
                x_t = x_seq[:, t]
                x_tp1 = x_seq[:, t + 1]
                u_t = u_seq[:, t, :]

                # Encode current frame
                mu_z_t, logvar_z_t = model.vae.encoder(x_t)
                z_t = model.vae.reparameterize(mu_z_t, logvar_z_t)

                # Reconstruct current frame
                x_recons_t, _ = model.vae.decoder(z_t)
                L_x_t = recons_loss(x_t.view(B, -1), x_recons_t.view(B, -1))
                mse_x_t = mse_loss(x_t.view(B, -1), x_recons_t.view(B, -1))

                # Predict next frame
                if hasattr(model.transition, "gru_step"):
                    # RNN model
                    mu_z_t_hat, logvar_z_t_hat = model.transition.prior(h_t)
                    KL_t = kl_diag_gaussian(
                        mu_z_t, logvar_z_t, mu_z_t_hat, logvar_z_t_hat
                    )

                    h_t = model.transition.gru_step(z_t, u_t, h_t)
                    mu_z_tp1_hat, logvar_z_tp1_hat = model.transition.prior(h_t)
                    z_tp1_hat = model.vae.reparameterize(
                        mu_z_tp1_hat, logvar_z_tp1_hat
                    )
                else:
                    # MLP model
                    KL_t = kl_diag_gaussian(
                        mu_z_t,
                        logvar_z_t,
                        torch.zeros_like(mu_z_t),
                        torch.zeros_like(logvar_z_t),
                    )

                    A_t, B_t, o_t, v_t, r_t = model.transition(z_t)
                    z_tp1_hat = model.transition.forward_dynamics(
                        A_t, B_t, o_t, z_t, u_t
                    )

                x_pred_tp1, _ = model.vae.decoder(z_tp1_hat)
                L_x_tp1_t = recons_loss(
                    x_tp1.view(B, -1), x_pred_tp1.view(B, -1)
                )
                mse_tp1 = mse_loss(x_tp1.view(B, -1), x_pred_tp1.view(B, -1))

                seq_L_x.append(L_x_t.cpu().numpy())
                seq_L_x_tp1.append(L_x_tp1_t.cpu().numpy())
                seq_KL.append(KL_t.cpu().numpy())
                seq_MSE_x.append(mse_x_t.cpu().numpy())
                seq_MSE_x_tp1.append(mse_tp1.cpu().numpy())

            all_L_x.append(np.stack(seq_L_x, axis=1))
            all_L_x_tp1.append(np.stack(seq_L_x_tp1, axis=1))
            all_KL.append(np.stack(seq_KL, axis=1))
            all_MSE_x.append(np.stack(seq_MSE_x, axis=1))
            all_MSE_x_tp1.append(np.stack(seq_MSE_x_tp1, axis=1))

    all_L_x = np.concatenate(all_L_x, axis=0)
    all_L_x_tp1 = np.concatenate(all_L_x_tp1, axis=0)
    all_KL = np.concatenate(all_KL, axis=0)
    all_MSE_x = np.concatenate(all_MSE_x, axis=0)
    all_MSE_x_tp1 = np.concatenate(all_MSE_x_tp1, axis=0)

    metrics = {
        "per_timestep": {
            "L_x_mean": all_L_x.mean(axis=0).tolist(),
            "L_x_std": all_L_x.std(axis=0).tolist(),
            "L_x_tp1_mean": all_L_x_tp1.mean(axis=0).tolist(),
            "L_x_tp1_std": all_L_x_tp1.std(axis=0).tolist(),
            "KL_mean": all_KL.mean(axis=0).tolist(),
            "KL_std": all_KL.std(axis=0).tolist(),
            "MSE_x_mean": all_MSE_x.mean(axis=0).tolist(),
            "MSE_x_std": all_MSE_x.std(axis=0).tolist(),
            "MSE_x_tp1_mean": all_MSE_x_tp1.mean(axis=0).tolist(),
            "MSE_x_tp1_std": all_MSE_x_tp1.std(axis=0).tolist(),
        },
        "overall": {
            "L_x_mean": float(all_L_x.mean()),
            "L_x_tp1_mean": float(all_L_x_tp1.mean()),
            "KL_mean": float(all_KL.mean()),
            "MSE_x_mean": float(all_MSE_x.mean()),
            "MSE_x_tp1_mean": float(all_MSE_x_tp1.mean()),
            "total_loss_mean": float(
                all_L_x.mean() + all_L_x_tp1.mean() + e2c_config.beta_kl * all_KL.mean()
            ),
        },
        "num_sequences": all_L_x.shape[0],
        "sequence_length": all_L_x.shape[1],
    }

    return metrics


def _prepare_img(img: torch.Tensor):
    arr = img.detach().cpu().numpy()
    if arr.ndim == 3:
        arr = arr[-1]
    # fill NaNs with zero (black)
    return np.nan_to_num(arr, nan=0.0)


def evaluate_multi_horizon_mse(
    model,
    test_loader,
    device: str,
    horizons: List[int],
) -> Dict[str, Dict[str, float]]:

    compute_entire = False
    numeric_horizons: List[int] = []
    for h in horizons:
        if isinstance(h, str) and h == "entire":
            compute_entire = True
        else:
            numeric_horizons.append(int(h))
    numeric_horizons = sorted(set([h for h in numeric_horizons if h >= 0]))

    acc: Dict[str, List[float]] = {str(h): [] for h in numeric_horizons}
    if compute_entire:
        acc["entire"] = []

    with torch.inference_mode():
        for x_seq, u_seq in tqdm(test_loader, desc="Horizon MSE"):
            x_seq = x_seq.to(device)  # (B, K+1, C, H, W)
            u_seq = u_seq.to(device)  # (B, K, u_dim)

            B, Kp1, _, _, _ = x_seq.size()
            K = Kp1 - 1  # number of actions/steps available

            rnn_mode = hasattr(model.transition, "gru_step")
            h_t_main = None
            if rnn_mode and hasattr(model.transition, "init_state"):
                h_t_main = model.transition.init_state(B, device)

            for t in range(K):
                x_t = x_seq[:, t]
                u_t = u_seq[:, t, :]

                # encode posterior at time t
                mu_z_t, logvar_z_t = model.vae.encoder(x_t)
                z_t = model.vae.reparameterize(mu_z_t, logvar_z_t)

                # horizon 0 (reconstruction)
                if 0 in numeric_horizons:
                    x_recons_t, _ = model.vae.decoder(z_t)
                    mse0 = mse_loss(x_t.view(B, -1), x_recons_t.view(B, -1))
                    acc["0"].extend(mse0.detach().cpu().numpy().tolist())

                # get recurrent state at time t before stepping main state
                if rnn_mode:
                    h_local = h_t_main

                max_h = max([h for h in numeric_horizons if h > 0], default=0)
                if max_h > 0:
                    max_valid_h = min(max_h, K - t)
                    z_curr = z_t
                    if rnn_mode:
                        h_curr = h_local
                    step_idx = t
                    for step in range(1, max_valid_h + 1):
                        if rnn_mode:
                            # prior before step is parameterized by h_curr after updating with (z_curr, u)
                            u_curr = u_seq[:, step_idx, :]
                            h_tp1 = model.transition.gru_step(z_curr, u_curr, h_curr)
                            mu_tp1, logvar_tp1 = model.transition.prior(h_tp1)
                            z_tp1 = model.vae.reparameterize(mu_tp1, logvar_tp1)
                            h_curr = h_tp1
                        else:
                            # MLP
                            u_curr = u_seq[:, step_idx, :]
                            A_t, B_t, o_t, v_t, r_t = model.transition(z_curr)
                            z_tp1 = model.transition.forward_dynamics(A_t, B_t, o_t, z_curr, u_curr)

                        # decode and compute mse if this step matches
                        if step in numeric_horizons:
                            x_pred, _ = model.vae.decoder(z_tp1)
                            x_target = x_seq[:, t + step]
                            mse_h = mse_loss(x_target.view(B, -1), x_pred.view(B, -1))
                            acc[str(step)].extend(mse_h.detach().cpu().numpy().tolist())

                        z_curr = z_tp1
                        step_idx += 1

                if rnn_mode:
                    h_t_main = model.transition.gru_step(z_t, u_t, h_t_main)

            if compute_entire:
                if rnn_mode and hasattr(model.transition, "init_state"):
                    h_entire = model.transition.init_state(B, device)
                else:
                    h_entire = None

                x0 = x_seq[:, 0]
                mu0, logvar0 = model.vae.encoder(x0)
                z_curr = model.vae.reparameterize(mu0, logvar0)

                per_sample_mses: List[torch.Tensor] = []
                step_idx = 0
                for step in range(1, Kp1):  # steps to predict: 1..K
                    if rnn_mode:
                        u_curr = u_seq[:, step_idx, :]
                        h_tp1 = model.transition.gru_step(z_curr, u_curr, h_entire)
                        mu_tp1, logvar_tp1 = model.transition.prior(h_tp1)
                        z_tp1 = model.vae.reparameterize(mu_tp1, logvar_tp1)
                        h_entire = h_tp1
                    else:
                        u_curr = u_seq[:, step_idx, :]
                        A_t, B_t, o_t, v_t, r_t = model.transition(z_curr)
                        z_tp1 = model.transition.forward_dynamics(A_t, B_t, o_t, z_curr, u_curr)

                    x_pred, _ = model.vae.decoder(z_tp1)
                    x_target = x_seq[:, step]  # t = step
                    mse_step = mse_loss(x_target.view(B, -1), x_pred.view(B, -1))  # (B,)
                    per_sample_mses.append(mse_step)

                    z_curr = z_tp1
                    step_idx += 1

                if per_sample_mses:
                    traj_mean = torch.stack(per_sample_mses, dim=0).mean(dim=0)  # (B,)
                    acc["entire"].extend(traj_mean.detach().cpu().numpy().tolist())

    # Reduce
    out: Dict[str, Dict[str, float]] = {}
    for key, vals in acc.items():
        if len(vals) == 0:
            out[key] = {"mean": float("nan"), "std": float("nan"), "count": 0}
            continue
        arr = np.array(vals, dtype=np.float64)
        finite = np.isfinite(arr)
        arr = arr[finite]
        if arr.size == 0:
            out[key] = {"mean": float("nan"), "std": float("nan"), "count": 0}
        else:
            out[key] = {
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "count": int(arr.size),
            }
    return out


def _collect_horizon_frames(
    model,
    x_seq: torch.Tensor,
    u_seq: torch.Tensor,
    horizons: List[int],
    device: str,
):
    frames = {}
    with torch.no_grad():
        x0 = x_seq[:, 0]
        mu0, logvar0 = model.vae.encoder(x0)
        z_curr = model.vae.reparameterize(mu0, logvar0)
        x_recon0, _ = model.vae.decoder(z_curr)
        frames[0] = (x0.detach(), x_recon0.detach())

        rnn_mode = hasattr(model.transition, "gru_step")
        if rnn_mode and hasattr(model.transition, "init_state"):
            h_curr = model.transition.init_state(x_seq.size(0), device)
        else:
            h_curr = None

        max_h = max([h for h in horizons if h > 0], default=0)
        for step in range(1, max_h + 1):
            u_curr = u_seq[:, step - 1, :]
            if rnn_mode:
                h_tp1 = model.transition.gru_step(z_curr, u_curr, h_curr)
                mu_tp1, logvar_tp1 = model.transition.prior(h_tp1)
                z_tp1 = model.vae.reparameterize(mu_tp1, logvar_tp1)
                h_curr = h_tp1
            else:
                A_t, B_t, o_t, v_t, r_t = model.transition(z_curr)
                z_tp1 = model.transition.forward_dynamics(A_t, B_t, o_t, z_curr, u_curr)

            if step in horizons:
                x_pred, _ = model.vae.decoder(z_tp1)
                frames[step] = (x_seq[:, step].detach(), x_pred.detach())

            z_curr = z_tp1

    return frames


def visualize_reconstructions(
    model,
    batch: Tuple[torch.Tensor, torch.Tensor],
    device: str,
    save_path: Path,
    num_examples: int = 4,
    title: Optional[str] = None,
):
    if num_examples <= 0:
        return

    horizons = [0, 1, 5, 20, 100]

    x_seq, u_seq = batch
    x_seq = x_seq.to(device)
    u_seq = u_seq.to(device)

    K = x_seq.size(1) - 1  # available steps
    valid_horizons = [h for h in horizons if h == 0 or h <= K]
    missing = [h for h in horizons if h not in valid_horizons]

    num = min(num_examples, x_seq.size(0))
    rows = num * 2
    width_per_col = 3.0
    height_per_row = 2.5
    title_fontsize = 18
    label_fontsize = 12
    fig, axes = plt.subplots(
        rows,
        len(valid_horizons),
        figsize=(width_per_col * len(valid_horizons), height_per_row * rows),
    )
    if title:
        fig.suptitle(title, fontsize=title_fontsize)
    axes = np.atleast_2d(axes).reshape(rows, len(valid_horizons))

    for i in range(num):
        frames = _collect_horizon_frames(
            model, x_seq[i : i + 1], u_seq[i : i + 1], valid_horizons, device
        )
        top_row = 2 * i
        bottom_row = top_row + 1

        for col, h in enumerate(valid_horizons):
            target, pred = frames.get(h, (None, None))

            ax_target = axes[top_row, col]
            ax_pred = axes[bottom_row, col]
            ax_target.axis("off")
            ax_pred.axis("off")

            if target is not None:
                ax_target.imshow(_prepare_img(target[0]), cmap="gray")
            else:
                ax_target.text(0.5, 0.5, "N/A", ha="center", va="center")

            if pred is not None:
                ax_pred.imshow(_prepare_img(pred[0]), cmap="gray")
            else:
                ax_pred.text(0.5, 0.5, "N/A", ha="center", va="center")

            if i == 0:
                title = "0 (gt/recon)" if h == 0 else f"{h} (next)"
                ax_target.set_title(title, fontsize=label_fontsize)

            if col == 0:
                ax_target.set_ylabel("x_{t+h}", rotation=90, fontsize=label_fontsize)
                ax_pred.set_ylabel("pred(x_{t+h})", rotation=90, fontsize=label_fontsize)

    fig.tight_layout(rect=(0, 0, 1, 0.95) if title else None)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def create_reconstruction_gif(
    model,
    batch: Tuple[torch.Tensor, torch.Tensor],
    device: str,
    save_path: Path,
    max_steps: int = 10,
    fps: int = 4,
):
    x_seq, u_seq = batch
    x_seq = x_seq.to(device)
    u_seq = u_seq.to(device)

    B, Kp1, _, _, _ = x_seq.shape
    steps = min(Kp1 - 1, max_steps)
    if steps <= 0:
        return

    def to_vis(img: torch.Tensor):
        arr = img.detach().cpu().numpy()
        if arr.shape[0] > 1:
            arr = arr[0]
        else:
            arr = arr[0]
        amin, amax = arr.min(), arr.max()
        if amax > amin:
            arr = (arr - amin) / (amax - amin)
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        return np.stack([arr] * 3, axis=-1)

    frames = []
    h_t = None
    if hasattr(model.transition, "init_state"):
        h_t = model.transition.init_state(1, device)

    for t in range(steps):
        x_t = x_seq[:1, t]
        x_tp1 = x_seq[:1, t + 1]
        u_t = u_seq[:1, t, :]

        with torch.no_grad():
            mu_z_t, logvar_z_t = model.vae.encoder(x_t)
            z_t = model.vae.reparameterize(mu_z_t, logvar_z_t)
            x_recons_t, _ = model.vae.decoder(z_t)

            if hasattr(model.transition, "gru_step"):
                mu_z_t_hat, logvar_z_t_hat = model.transition.prior(h_t)
                h_t = model.transition.gru_step(z_t, u_t, h_t)
                mu_z_tp1_hat, logvar_z_tp1_hat = model.transition.prior(h_t)
                z_tp1_hat = model.vae.reparameterize(
                    mu_z_tp1_hat, logvar_z_tp1_hat
                )
            else:
                A_t, B_t, o_t, v_t, r_t = model.transition(z_t)
                z_tp1_hat = model.transition.forward_dynamics(
                    A_t, B_t, o_t, z_t, u_t
                )

            x_pred_tp1, _ = model.vae.decoder(z_tp1_hat)

        vis = [
            to_vis(x_t[0]),
            to_vis(x_recons_t[0]),
            to_vis(x_tp1[0]),
            to_vis(x_pred_tp1[0]),
        ]
        row = np.concatenate(vis, axis=1)
        frames.append(row)

    if len(frames) > 1:
        duration = 1.0 / fps
        imageio.mimsave(save_path, frames, duration=duration, loop=0)


def plot_reconstruction_losses(
    metrics: dict, save_path: Path, title: Optional[str] = None
):
    per_ts = metrics.get("per_timestep", {})
    steps = np.arange(len(per_ts.get("L_x_mean", [])))
    if steps.size == 0:
        return

    mse_x_mean = np.array(per_ts.get("MSE_x_mean", []))
    mse_x_std = np.array(per_ts.get("MSE_x_std", []))
    mse_tp1_mean = np.array(per_ts.get("MSE_x_tp1_mean", []))
    mse_tp1_std = np.array(per_ts.get("MSE_x_tp1_std", []))

    n = metrics.get("num_sequences", 0)
    def ci_band(std_arr: np.ndarray) -> np.ndarray:
        if n and n > 1:
            return 1.96 * std_arr / np.sqrt(n)
        return std_arr

    fig, ax = plt.subplots(figsize=(7, 4))

    if mse_x_mean.size:
        ax.plot(steps, mse_x_mean, label="recon MSE")
        ci = ci_band(mse_x_std)
        ax.fill_between(steps, mse_x_mean - ci, mse_x_mean + ci, alpha=0.2)
    if mse_tp1_mean.size:
        ax.plot(steps, mse_tp1_mean, label="next MSE")
        ci = ci_band(mse_tp1_std)
        ax.fill_between(steps, mse_tp1_mean - ci, mse_tp1_mean + ci, alpha=0.2)

    ax.set_xlabel("timestep")
    ax.set_ylabel("loss")
    if title:
        ax.set_title(title)
    handles, labels = ax.get_legend_handles_labels()
    ci_handle = Line2D(
        [0],
        [0],
        color="gray",
        marker="_",
        linestyle="none",
        markersize=10,
        label="95% CI band",
    )
    handles.append(ci_handle)
    labels.append("95% CI band")
    ax.legend(handles, labels)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def aggregate_metrics(metrics_list: List[dict]) -> Dict:
    """Aggregate a list of metrics dicts (mean/std across runs)."""
    if not metrics_list:
        return {}

    keys_ts = metrics_list[0]["per_timestep"].keys()
    keys_overall = metrics_list[0]["overall"].keys()

    agg = {
        "per_timestep": {},
        "overall": {},
        "num_runs": len(metrics_list),
    }

    # stack per-run means
    for key in keys_ts:
        runs = [np.array(m["per_timestep"][key]) for m in metrics_list]
        mat = np.stack(runs, axis=0)
        agg["per_timestep"][key] = mat.mean(axis=0).tolist()
        agg["per_timestep"][f"{key}_std_across_runs"] = mat.std(axis=0).tolist()

    # overall
    for key in keys_overall:
        vals = np.array([m["overall"][key] for m in metrics_list], dtype=float)
        agg["overall"][key] = float(vals.mean())
        agg["overall"][f"{key}_std_across_runs"] = float(vals.std())

    # metainfo
    agg["num_sequences"] = metrics_list[0].get("num_sequences")
    agg["sequence_length"] = metrics_list[0].get("sequence_length")
    return agg


def evaluate_checkpoint(
    checkpoint_path: Path,
    device: str,
    num_visuals: int,
    save_visuals: bool,
    max_rollout_gifs: int = 0,
    experiment_name: Optional[str] = None,
    seed: Optional[int] = None,
):
    exp_from_path, seed_from_path, model_tag, timestamp = extract_run_metadata(checkpoint_path)
    exp_name = experiment_name or exp_from_path
    seed_val = seed if seed is not None else seed_from_path

    model, model_tag, checkpoint = load_model_from_checkpoint(
        str(checkpoint_path), device
    )
    print(f"Experiment: {exp_name} | Seed: {seed_val if seed_val is not None else 'unknown'}")
    print(f"Model tag: {model_tag}")
    print(f"Epoch: {checkpoint['epoch'] + 1}")

    test_loader = create_mujoco_dataloader(split="test", seq_len="full")
    sample_batch = next(iter(test_loader))
    print(f"Test sequences: {len(test_loader.dataset)}\n")

    metrics = evaluate_rollout(model, test_loader, device)

    # multi-horizon MSEs
    horizon_stats = evaluate_multi_horizon_mse(
        model, test_loader, device, horizons=[0, 1, 5, 20, 100, "entire"]
    )
    metrics["horizon_mse"] = horizon_stats

    results_dir = create_run_dir(
        model_tag,
        base_dir="results",
        timestamp=str(timestamp),
        experiment_name=exp_name,
        seed=seed_val,
    )

    metrics_path = results_dir / f"metrics_epoch_{checkpoint['epoch']+1:03d}.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    if save_visuals:
        vae_label, transition_label = format_model_label(model_tag)
        viz_title = f"{format_experiment_label(exp_name)}, {vae_label}, {transition_label} Image Reconstructions"
        loss_title = format_plot_title(
            experiment_name=exp_name, model_tag=model_tag, epoch=checkpoint["epoch"] + 1
        )
        viz_path = results_dir / f"recon_epoch_{checkpoint['epoch']+1:03d}.png"
        loss_path = results_dir / f"recon_losses_epoch_{checkpoint['epoch']+1:03d}.png"
        gif_path = results_dir / f"recon_epoch_{checkpoint['epoch']+1:03d}.gif"
        visualize_reconstructions(
            model, sample_batch, device, viz_path, num_visuals, title=viz_title
        )
        plot_reconstruction_losses(metrics, loss_path, title=loss_title)
        create_reconstruction_gif(model, sample_batch, device, gif_path)
        print(f"Saved visuals to: {viz_path.name}, {loss_path.name}, {gif_path.name}")

    if save_visuals and max_rollout_gifs != 0:
        try:
            from visualize_rnn import plot_rollout_mujoco as plot_rollout_rnn
            from visualize_mlp import plot_rollout_mujoco as plot_rollout_mlp

            gifs_dir = results_dir / "gifs_test_rollouts"
            gifs_dir.mkdir(parents=True, exist_ok=True)
            ds = test_loader.dataset 
            frames_np = ds.frames  # (N,T,1,H,W)
            actions_np = ds.actions  # (N,T,A)
            num_traj = frames_np.shape[0]
            max_traj = num_traj if max_rollout_gifs < 0 else min(num_traj, max_rollout_gifs)

            if hasattr(model.transition, "gru_step"):
                print(f"Creating RNN rollout GIFs for {max_traj} test trajectories...")
                for traj_id in tqdm(range(max_traj), desc="GIFs"):
                    gif_path = gifs_dir / f"traj_{traj_id:04d}.gif"
                    plot_rollout_rnn(
                        model=model,
                        frames=frames_np,
                        actions=actions_np,
                        traj_id=traj_id,
                        t0=0,
                        horizon=None,
                        make_gif=True,
                        gif_path=str(gif_path),
                        show_plots=False,
                        device=device,
                    )
            else:
                print(f"Creating MLP rollout GIFs for {max_traj} test trajectories...")
                for traj_id in tqdm(range(max_traj), desc="GIFs"):
                    gif_path = gifs_dir / f"traj_{traj_id:04d}.gif"
                    plot_rollout_mlp(
                        model=model,
                        frames=frames_np,
                        actions=actions_np,
                        traj_id=traj_id,
                        t0=0,
                        horizon=None,
                        make_gif=True,
                        gif_path=str(gif_path),
                        show_plots=False,
                        device=device,
                    )
        except Exception as e:
            print(f"Failed to create test rollout GIFs: {e}")

    print("\nResults:")
    print(f"Overall L_x: {metrics['overall']['L_x_mean']:.4f}")
    print(f"Overall L_x_tp1: {metrics['overall']['L_x_tp1_mean']:.4f}")
    print(f"Overall KL: {metrics['overall']['KL_mean']:.4f}")
    print(f"Overall MSE: {metrics['overall']['MSE_x_mean']:.6f} / {metrics['overall']['MSE_x_tp1_mean']:.6f}")
    print(f"Total loss: {metrics['overall']['total_loss_mean']:.4f}")
    print(f"Metrics saved to: {metrics_path}\n")

    return metrics_path, metrics


def find_latest_checkpoints(
    model_tag: str,
    experiment_name: str,
    seeds: Optional[List[int]],
    runs_per_seed: int,
) -> List[Tuple[Optional[int], Path]]:
    exp_dir = Path("checkpoints") / experiment_name
    if not exp_dir.exists():
        return []

    if seeds:
        seed_dirs = [exp_dir / seed_dir_name(s) for s in seeds]
    else:
        seed_dirs = [d for d in exp_dir.iterdir() if d.is_dir()]

    checkpoints: List[Tuple[Optional[int], Path]] = []
    for seed_dir in seed_dirs:
        if not seed_dir.exists():
            continue
        model_dir = seed_dir / model_tag
        if not model_dir.exists():
            continue
        run_dirs = sorted([d for d in model_dir.iterdir() if d.is_dir()])
        collected = 0
        for run_dir in reversed(run_dirs):
            ckpt = get_latest_checkpoint(run_dir)
            if ckpt is not None:
                seed_val = parse_seed_dir(seed_dir.name)
                checkpoints.append((seed_val, ckpt))
                collected += 1
            if collected >= runs_per_seed:
                break
    return checkpoints


def main():
    parser = argparse.ArgumentParser(description="Evaluate E2C model(s) on the test set")
    parser.add_argument(
        "checkpoint",
        nargs="?",
        help="Path to checkpoint file. If omitted, latest checkpoints for model-types are used.",
    )
    parser.add_argument(
        "--config",
        dest="config_module",
        type=str,
        help="Config module name or dotted path (determines experiment folder naming).",
    )
    parser.add_argument(
        "--model-types",
        nargs="+",
        choices=["mlp", "rnn"],
        help="Dynamics model types to evaluate when checkpoint is not provided (default: both).",
    )
    parser.add_argument(
        "--num-visuals",
        type=int,
        default=4,
        help="Number of sequences to visualize in reconstruction grids.",
    )
    parser.add_argument(
        "--skip-visuals",
        action="store_true",
        help="Disable saving reconstruction images and loss plots.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of latest runs per seed to evaluate/aggregate.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        help="Seed subfolders to evaluate (default: all seeds found for the experiment).",
    )
    parser.add_argument(
        "--max-rollout-gifs",
        type=int,
        default=0,
        help="# of test trajectories to render as rollout gifs (0 to skip, -1 for all).",
    )
    args = parser.parse_args()

    config_module_path = os.environ.get("E2C_CONFIG", "configs.base_config")
    if args.config_module:
        module = args.config_module
        if not module.startswith("configs."):
            module = f"configs.{module}"
        os.environ["E2C_CONFIG"] = module
        config_module_path = module
    elif not config_module_path.startswith("configs."):
        config_module_path = f"configs.{config_module_path}"
        os.environ["E2C_CONFIG"] = config_module_path

    try:
        import config as config_module

        new_cfg = config_module._load_config(config_module_path)
        current_cfg = config_module.e2c_config
        current_cfg.__dict__.clear()
        for key in dir(new_cfg):
            if key.startswith("_"):
                continue
            val = getattr(new_cfg, key)
            if callable(val):
                continue
            setattr(current_cfg, key, val)
        globals()["e2c_config"] = current_cfg
    except Exception as e:
        print(f"Failed to reload config '{config_module_path}': {e}")

    experiment_name = config_module_path.split(".")[-1]
    device = get_device()

    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            print(f"Checkpoint not found: {checkpoint_path}")
            return
        exp_from_path, seed_from_path, _, _ = extract_run_metadata(checkpoint_path)
        eval_experiment = exp_from_path or experiment_name
        evaluate_checkpoint(
            checkpoint_path,
            device,
            args.num_visuals,
            not args.skip_visuals,
            args.max_rollout_gifs,
            experiment_name=eval_experiment,
            seed=seed_from_path,
        )
        return

    model_types = args.model_types or ["mlp", "rnn"]
    vae_types = ["conv"]
    seeds = args.seeds

    for model_type in model_types:
        for vae_type in vae_types:
            model_tag = f"{model_type}-{vae_type}"
            ckpts = find_latest_checkpoints(model_tag, experiment_name, seeds, args.runs)
            if not ckpts:
                print(f"No checkpoint found for experiment '{experiment_name}', model tag '{model_tag}'.")
                continue

            metrics_list = []
            for seed_val, ckpt in ckpts:
                seed_msg = f"seed={seed_val}" if seed_val is not None else "seed=unknown"
                print(f"\nEvaluating {model_tag.upper()} checkpoint ({seed_msg}): {ckpt} ===\n")
                _, metrics = evaluate_checkpoint(
                    ckpt,
                    device,
                    args.num_visuals,
                    not args.skip_visuals,
                    args.max_rollout_gifs,
                    experiment_name=experiment_name,
                    seed=seed_val,
                )
                metrics_list.append(metrics)

            if len(metrics_list) > 1:
                agg = aggregate_metrics(metrics_list)
                agg_dir = (
                    Path("results")
                    / experiment_name
                    / model_tag
                    / f"aggregate_{len(metrics_list)}runs"
                )
                agg_dir.mkdir(parents=True, exist_ok=True)
                agg_path = agg_dir / "metrics_agg.json"
                with open(agg_path, "w") as f:
                    json.dump(agg, f, indent=2)
                agg_title = format_plot_title(
                    experiment_name=experiment_name,
                    model_tag=model_tag,
                    runs=len(metrics_list),
                )
                plot_reconstruction_losses(
                    agg, agg_dir / "recon_losses_agg.png", title=agg_title
                )
                print(f"\nAggregated {len(metrics_list)} runs for {model_tag}: {agg_path}")


if __name__ == "__main__":
    main()
