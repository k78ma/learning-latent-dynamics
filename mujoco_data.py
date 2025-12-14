import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from config import e2c_config

class MujocoSequenceDataset(Dataset):
    """    
    x_seq: k+1, stacked frames (grayscale), h, w
    u_seq: k, control dim
    """
    
    def __init__(
        self,
        split: str = None,
        seq_len: int | str | None = None,
    ):
        super().__init__()
        self.data_dir = e2c_config.data_dir
        self.split = split
        self.stack_frames = e2c_config.state_seq_len
        
        # load appropriate files based on split
        if split is None:
            frames_file = e2c_config.frames_filename
            actions_file = e2c_config.actions_filename
        else:
            frames_file = f"{split}_{e2c_config.frames_filename}"
            actions_file = f"{split}_{e2c_config.actions_filename}"
        
        frames_path = os.path.join(self.data_dir, frames_file)
        actions_path = os.path.join(self.data_dir, actions_file)

        self.frames = np.load(frames_path)    # (N, T, 1, H, W)
        self.actions = np.load(actions_path)  # (N, T, A)
        
        self.N, self.T, _, self.H, self.W = self.frames.shape
        _, _, self.u_dim = self.actions.shape

        if seq_len is None:
            self.seq_len = e2c_config.rnn_seq_len
        elif isinstance(seq_len, str) and seq_len == "full":
            self.seq_len = self.T - self.stack_frames
        else:
            self.seq_len = int(seq_len)
        if self.seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {self.seq_len}")
        
        total_frames_needed = self.seq_len + self.stack_frames
        if total_frames_needed > self.T:
            raise ValueError(f"seq_len + stack_frames = {total_frames_needed} > T={self.T}")
        
        # pre-compute all valid (traj, start_time) pairs
        # need stack_frames-1 frames before for initial stacking
        self.valid_indices = []
        for traj_id in range(self.N):
            for t0 in range(self.stack_frames - 1, self.T - self.seq_len):
                self.valid_indices.append((traj_id, t0))
        
        print(f"[MujocoSequenceDataset] {len(self.valid_indices)} sequences available (seq_len={self.seq_len})")
        print(f"  Frames shape: {self.frames.shape}, Actions shape: {self.actions.shape}")
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        traj_id, t0 = self.valid_indices[idx]
        
        x_seq = np.zeros((self.seq_len + 1, self.stack_frames, self.H, self.W), dtype=np.float32)
        
        for t in range(self.seq_len + 1):
            frame_start = t0 + t - (self.stack_frames - 1)
            frame_end = t0 + t + 1
            
            if frame_start < 0:
                raise ValueError(f"frame_start={frame_start} < 0 (traj_id={traj_id}, t0={t0}, t={t}, stack_frames={self.stack_frames})")
            if frame_end > self.T:
                raise ValueError(f"frame_end={frame_end} > T={self.T} (traj_id={traj_id}, t0={t0}, t={t})")
            
            stacked = self.frames[traj_id, frame_start:frame_end, 0]  # (stack_frames, H, W)
            
            if stacked.shape[0] != self.stack_frames:
                raise ValueError(f"stacked.shape={stacked.shape} but expected ({self.stack_frames}, {self.H}, {self.W}). "
                                f"frame_start={frame_start}, frame_end={frame_end}, traj_id={traj_id}, t0={t0}, t={t}")
            
            x_seq[t] = stacked
        
        u_seq = self.actions[traj_id, t0:t0 + self.seq_len, :]  # (seq_len, A)
        
        return torch.from_numpy(x_seq).float(), torch.from_numpy(u_seq).float()


def create_mujoco_dataloader(split: str, seq_len: int | str | None = None):
    if split not in ["train", "val", "test"]:
        raise ValueError(f"split must be one of ['train', 'val', 'test'], got {split}")
    
    dataset = MujocoSequenceDataset(split=split, seq_len=seq_len)
    
    batch_size = e2c_config.batch_size
    num_workers = os.cpu_count()
    
    shuffle = (split == "train")
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return loader
