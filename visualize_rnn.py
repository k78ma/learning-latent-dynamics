import os
import numpy as np
import matplotlib.pyplot as plt
import torch

from models.e2c.rnn_e2c import RecurrentE2C
from models.e2c.common import sample_normal
# from models.e2c.mlp_e2c import MLP_E2C
from config import e2c_config
import imageio.v2 as imageio  # add this

def load_mujoco_data(data_dir: str = "data/raw"):
    frames_path = os.path.join(data_dir, "frames.npy")
    actions_path = os.path.join(data_dir, "actions.npy")

    frames = np.load(frames_path)   # (N, T, 1, H, W)
    actions = np.load(actions_path) # (N, T, A)
    return frames, actions

def main(
    ckpt_path: str = "mujoco-ckpt-recurrent/recurrent_e2c_15000.pt",
    data_dir: str = "data/raw",
    num_recon_samples: int = 5,
    traj_for_latent: int = None,
):
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    frames, actions = load_mujoco_data(data_dir)
    # frames: (N, T, 1, H, W), actions: (N, T, A)
    _, _, _, H, W = frames.shape
    _, _, A = actions.shape
    x_dim = H * W
    u_dim = A

    model = RecurrentE2C(x_dim, u_dim)
    model.load_state_dict(state_dict)
    model.eval()

    device = "cpu"
    model.to(device)

    print(f"[viz] frames.shape  = {frames.shape}")
    print(f"[viz] actions.shape = {actions.shape}")

    plot_latent_trajectory(
        model=model,
        frames=frames,
        traj_id=traj_for_latent,
        device=device,
    )

    plot_rollout_mujoco(
        model=model,
        frames=frames,
        traj_id=traj_for_latent,
        actions=actions,
        t0=0,
        horizon=None,
        make_gif=True,
        gif_path="rollout.gif",
        device=device,
    )

def plot_latent_trajectory(model, frames, traj_id=None, device="cpu"):
    model.eval()
    device = str(device)
    model.to(device)

    N, T, _, H, W = frames.shape
    S = e2c_config.state_seq_len

    if traj_id is None:
        traj_id = np.random.randint(0, N)
    assert 0 <= traj_id < N, "traj_id out of range"

    print(f"[viz] Plotting latent trajectory for traj_id={traj_id}")

    Z_list = []
    with torch.no_grad():
        for t in range(S - 1, T):
            stacked = frames[traj_id, t - (S - 1): t + 1, 0]  # (S, H, W)
            x_tensor = torch.from_numpy(stacked[None, :, :, :]).float().to(device)  # (1,S,H,W)
            mu, logvar = model.vae.encoder(x_tensor)  # (1, z_dim)
            mu_np = mu.cpu().numpy().squeeze(0)  # (z_dim,)
            Z_list.append(mu_np)

    Z = np.stack(Z_list, axis=0)  # (T_eff, z_dim)
    z_dim = Z.shape[1]
    T_eff = Z.shape[0]
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    if z_dim >= 2:
        sc = axs[0].scatter(Z[:, 0], Z[:, 1], c=np.arange(T_eff), cmap="viridis", s=5)
        axs[0].set_title("Latent Trajectory (z[0] vs z[1])")
        axs[0].set_xlabel("z[0]")
        axs[0].set_ylabel("z[1]")
        axs[0].axis("equal")
        cb = plt.colorbar(sc, ax=axs[0])
        cb.set_label("time step")
    else:
        axs[0].plot(np.arange(T_eff), Z[:, 0], "-o", markersize=2)
        axs[0].set_title("Latent z over time")
        axs[0].set_xlabel("t")
        axs[0].set_ylabel("z")

    for d in range(z_dim):
        axs[1].plot(np.arange(T_eff), Z[:, d], label=f"z[{d}]")
    axs[1].set_title("Latent components over time")
    axs[1].set_xlabel("time step")
    axs[1].legend()

    plt.tight_layout()
    plt.show()

def plot_rollout_mujoco(
    model,
    frames,
    actions,
    traj_id: int = None,
    t0: int = 0,
    horizon: int = None,
    make_gif: bool = False,
    gif_path: str = "rollout_gt_pred.gif",
    gif_fps: int = 4,
    show_plots: bool = True,
    save_plot_path: str | None = None,
    device: str = "cpu",
):
    model.eval()
    device = str(device)
    model.to(device)

    N, T, _, H, W = frames.shape
    _, _, A = actions.shape
    S = e2c_config.state_seq_len

    if traj_id is None:
        traj_id = np.random.randint(0, N)
    assert 0 <= traj_id < N

    t0 = int(t0) if t0 is not None else 0
    t0 = max(0, min(t0, T - 1))
    max_horizon = max(0, T - 1 - t0)
    if horizon is None or horizon > max_horizon:
        horizon = max_horizon

    print(f"[viz] Rollout: traj_id={traj_id}, t0={t0}, horizon={horizon}")

    with torch.no_grad():
        # init recurrent state
        h_t = model.transition.init_state(batch_size=1, device=device)

        stacked = frames[traj_id, t0 - (S - 1): t0 + 1, 0]  # (S, H, W)
        x_t_tensor = torch.from_numpy(stacked[None, :, :, :]).float().to(device)  # (1,S,H,W)
        mu_t, logvar_t = model.vae.encoder(x_t_tensor)  # (1, z_dim)
        z_t = model.vae.reparameterize(mu_t, logvar_t)  # (1, z_dim)

        gt_imgs = [frames[traj_id, t0, 0]]
        pred_imgs = [frames[traj_id, t0, 0]]

        for k in range(horizon):
            u_np = actions[traj_id, t0 + k, :]
            u_t = torch.from_numpy(u_np.reshape(1, A)).float().to(device)

            h_tp1 = model.transition.gru_step(z_t, u_t, h_t)
            mu_tp1_hat, logvar_tp1_hat = model.transition.prior(h_tp1)  # (1, z_dim) each
            sigma_tp1_hat = torch.exp(0.5 * logvar_tp1_hat)
            z_tp1 = sample_normal(mu_tp1_hat, sigma_tp1_hat)

            x_pred_tp1, _ = model.vae.decoder(z_tp1)  # (1,S,H,W)
            x_pred_img = x_pred_tp1.squeeze(0)[-1].cpu().numpy()  # (H, W)

            gt_imgs.append(frames[traj_id, t0 + k + 1, 0])
            pred_imgs.append(x_pred_img)

            z_t = z_tp1
            h_t = h_tp1

    T_vis = horizon + 1
    T_pic = min(10, T_vis)

    if show_plots or save_plot_path:
        fig, axs = plt.subplots(2, T_pic, figsize=(1.8 * T_vis, 4))

        for k in range(T_pic):
            axs[0, k].imshow(gt_imgs[k], cmap="gray", vmin=0.0, vmax=1.0)
            axs[0, k].set_title(f"GT t={t0 + k}")
            axs[0, k].axis("off")

            axs[1, k].imshow(pred_imgs[k], cmap="gray", vmin=0.0, vmax=1.0)
            axs[1, k].set_title(f"Pred t={t0 + k}")
            axs[1, k].axis("off")

        plt.suptitle(f"RecurrentE2C rollout (traj={traj_id}, start={t0}, shown={T_pic-1})")
        plt.tight_layout()
        if save_plot_path:
            plt.savefig(save_plot_path, dpi=200)
        if show_plots:
            plt.show()
        plt.close(fig)

    if make_gif:
        os.makedirs(os.path.dirname(gif_path) or ".", exist_ok=True)
        gif_frames = []
        for k in range(T_vis):
            gt = gt_imgs[k]
            pred = pred_imgs[k]

            frame = np.concatenate([gt, pred], axis=1)      # (H, 2W)
            frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8)

            frame_rgb = np.stack([frame, frame, frame], axis=2)  # (H, 2W, 3)
            gif_frames.append(frame_rgb)

        imageio.mimsave(gif_path, gif_frames, fps=gif_fps)
        print(f"[viz] Saved GIF to {gif_path}")



if __name__ == "__main__":
    main(
        ckpt_path="checkpoints/rnn-conv/20251212_140840/epoch_040.pt",
        data_dir="data/raw",
        num_recon_samples=5,
        traj_for_latent=67,
    )
