# sim/mujoco_data.py
import os
from typing import List, Tuple

import mujoco
import numpy as np

from config import e2c_config
from utils import set_random_seed
COLOR_WEIGHTS = np.array([0.299, 0.587, 0.114], dtype=np.float32)

def frames_to_grayscale(frames: np.ndarray) -> np.ndarray:
    gray = np.einsum("fhwc,c->fhw", frames, COLOR_WEIGHTS)
    gray = gray.astype(np.float32) / 255.0  # normalize
    gray = gray[:, None, :, :]  # (F, 1, H, W)
    return gray

def generate_single_trajectory() -> Tuple[np.ndarray, np.ndarray]:
    model = mujoco.MjModel.from_xml_path(e2c_config.mjcf_file)
    data = mujoco.MjData(model)

    action_dim = model.nu
    dt = 1.0 / e2c_config.fps
    total_steps = e2c_config.fps * e2c_config.traj_seconds

    frames: List[np.ndarray] = []
    actions: List[np.ndarray] = []

    with mujoco.Renderer(
        model,
        height=e2c_config.img_height,
        width=e2c_config.img_width,
    ) as renderer:
        # randomize initial joint configuration
        if e2c_config.randomize_init:
            for j in range(model.njnt):
                qadr = model.jnt_qposadr[j]
                jtype = model.jnt_type[j]
                if jtype == mujoco.mjtJoint.mjJNT_HINGE or jtype == mujoco.mjtJoint.mjJNT_SLIDE:
                    low, high = model.jnt_range[j]
                    if high > low:
                        data.qpos[qadr] = np.random.uniform(low, high)
                    else:
                        if jtype == mujoco.mjtJoint.mjJNT_HINGE:
                            data.qpos[qadr] = np.random.uniform(-np.pi, np.pi)
                        else:
                            data.qpos[qadr] = np.random.uniform(-0.1, 0.1)
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
        mujoco.mj_kinematics(model, data)

        for step_idx in range(total_steps):
            # random applied inputs
            u = np.random.uniform(low=-e2c_config.ctrl_max, high=e2c_config.ctrl_max, size=action_dim)
            data.ctrl[:] = u

            # step env until we reach next frame time
            target_time = (step_idx + 1) * dt
            while data.time < target_time:
                mujoco.mj_step(model, data)

            renderer.update_scene(data, camera=e2c_config.camera_name)
            frame = renderer.render()  # H x W x 3, uint8
            frames.append(frame)
            actions.append(u)

    frames = np.stack(frames, axis=0)
    actions = np.stack(actions, axis=0)
    frames_gray = frames_to_grayscale(frames)
    return frames_gray, actions

def save_npy(save_dir, frames, actions, indices, prefix:str = ""):
    frames_save_path = os.path.join(save_dir, f"{prefix}frames.npy")
    actions_save_path = os.path.join(save_dir, f"{prefix}actions.npy")
    np.save(frames_save_path, frames[indices])
    np.save(actions_save_path, actions[indices])
    print(f"[sim] Saved {prefix}frames to {frames_save_path}")
    print(f"[sim] Saved {prefix}actions to {actions_save_path}")

def collect_dataset(save_dir: str = "data/raw"):
    os.makedirs(save_dir, exist_ok=True)
    set_random_seed()

    all_frames = []
    all_actions = []

    for traj_idx in range(e2c_config.num_trajectories):
        print(f"[sim] Generating trajectory {traj_idx + 1}/{e2c_config.num_trajectories}")
        frames_gray, actions = generate_single_trajectory()
        all_frames.append(frames_gray)
        all_actions.append(actions)

    # concatenate along trajectory dimension
    frames_arr = np.stack(all_frames, axis=0)   # (N, T, 1, H, W)
    actions_arr = np.stack(all_actions, axis=0) # (N, T, A)

    indices = np.random.permutation(e2c_config.num_trajectories)

    split_at = [int(e2c_config.num_trajectories*0.8), int(e2c_config.num_trajectories*0.9)]
    train_indices, test_indices, val_indices = np.split(indices, split_at)
    
    save_npy(save_dir, frames_arr, actions_arr, train_indices, "train_")
    save_npy(save_dir, frames_arr, actions_arr, test_indices, "test_")
    save_npy(save_dir, frames_arr, actions_arr, val_indices, "val_")

    save_npy(save_dir, frames_arr, actions_arr, np.arange(len(frames_arr)), "")

    print(f"[sim] frames_arr.shape = {frames_arr.shape}, actions_arr.shape = {actions_arr.shape}")

if __name__ == "__main__":
    collect_dataset()