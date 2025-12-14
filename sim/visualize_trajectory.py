"""
python sim/visualize_trajectory.py --traj_idx 5 --save_gif traj5.gif --fps 30
"""

import argparse
import os

import numpy as np
import mediapy as media
import imageio.v2 as imageio


def load_trajectory(frames_path: str, actions_path: str | None, traj_idx: int):
    frames = np.load(frames_path) 
    print(f"[viz] Loaded frames from {frames_path}, shape={frames.shape}")

    if traj_idx < 0 or traj_idx >= frames.shape[0]:
        raise IndexError(f"traj_idx {traj_idx} out of range [0, {frames.shape[0]}).")

    traj_frames = frames[traj_idx]

    traj_actions = None
    if actions_path is not None and os.path.exists(actions_path):
        actions = np.load(actions_path)  # (N, T, A)
        print(f"[viz] Loaded actions from {actions_path}, shape={actions.shape}")
        traj_actions = actions[traj_idx]  # (T, A)

    return traj_frames, traj_actions


def to_rgb_video(traj_frames: np.ndarray) -> np.ndarray:

    if traj_frames.ndim != 4 or traj_frames.shape[1] != 1:
        raise ValueError(f"Expected frames of shape (T, 1, H, W), got {traj_frames.shape}")

    T, _, H, W = traj_frames.shape

    frames = traj_frames.astype(np.float32)

    if frames.max() <= 1.0 + 1e-6:
        frames = frames * 255.0

    frames = np.clip(frames, 0, 255).astype(np.uint8)

    # (T, 1, H, W) -> (T, H, W)
    frames = frames[:, 0, :, :]
    # grayscale to RGB by channel repetition
    frames_rgb = np.repeat(frames[..., None], 3, axis=-1)  # (T, H, W, 3)

    return frames_rgb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--frames_path",
        type=str,
        default="data/raw/frames.npy",
        help="Path to frames.npy (shape: (N, T, 1, H, W)).",
    )
    parser.add_argument(
        "--actions_path",
        type=str,
        default="data/raw/actions.npy",
        help="Path to actions.npy (shape: (N, T, A)). Optional.",
    )
    parser.add_argument(
        "--traj_idx",
        type=int,
        required=True,
        help="Index of trajectory to visualize (0-based).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second for video playback / saving.",
    )
    parser.add_argument(
        "--save_gif",
        type=str,
        default=None,
        help="If provided, save trajectory as a GIF to this path.",
    )

    args = parser.parse_args()

    traj_frames, traj_actions = load_trajectory(
        args.frames_path,
        args.actions_path if os.path.exists(args.actions_path) else None,
        args.traj_idx,
    )

    print(f"[viz] Trajectory {args.traj_idx}: frames shape={traj_frames.shape}")
    if traj_actions is not None:
        print(f"[viz] Trajectory {args.traj_idx}: actions shape={traj_actions.shape}")

    video = to_rgb_video(traj_frames)  # (T, H, W, 3)

    # SAVE GIF (imageio)
    if args.save_gif is not None:
        out_path = args.save_gif
        print(f"[viz] Saving GIF to {out_path}")
        # imageio expects a list of frames; duration is per-frame in seconds
        imageio.mimsave(out_path, list(video), duration=1.0 / args.fps)


if __name__ == "__main__":
    main()