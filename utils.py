import torch
from torch import nn
from config import e2c_config
import numpy as np
import os

def set_random_seed(seed: int = e2c_config.random_seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def get_device():
    device = torch.accelerator.current_accelerator(check_available=True) or "cpu"
    return device


## VAE
def conv_output_size(size: int, kernel_size: int = 4, stride: int = 2, padding: int = 1) -> int:
    return (size + 2 * padding - kernel_size) // stride + 1

def _as_list(value, num_layers: int):
    # broadcast scalar hyperparameters to list
    if isinstance(value, (list, tuple)):
        assert len(value) == num_layers, f"Expected list of length {num_layers}, got {len(value)}"
        return list(value)
    return [value] * num_layers

def get_vae_hidden_dims():
    h = e2c_config.img_height
    w = e2c_config.img_width

    conv_channels = getattr(e2c_config, "conv_channels", [32, 64, 128])
    num_layers = len(conv_channels)
    k = _as_list(getattr(e2c_config, "conv_kernel_size", 4), num_layers)
    s = _as_list(getattr(e2c_config, "conv_stride", 2), num_layers)
    p = _as_list(getattr(e2c_config, "conv_padding", 1), num_layers)

    for i in range(num_layers):
        h = conv_output_size(h, kernel_size=k[i], stride=s[i], padding=p[i])
        w = conv_output_size(w, kernel_size=k[i], stride=s[i], padding=p[i])

    last_c = conv_channels[-1]
    flatten_dim = last_c * h * w
    return h, w, flatten_dim

def get_vae_encoder_shapes():
    conv_channels = getattr(e2c_config, "conv_channels", [32, 64, 128])
    num_layers = len(conv_channels)
    k = _as_list(getattr(e2c_config, "conv_kernel_size", 4), num_layers)
    s = _as_list(getattr(e2c_config, "conv_stride", 2), num_layers)
    p = _as_list(getattr(e2c_config, "conv_padding", 1), num_layers)

    shapes = []
    h = e2c_config.img_height
    w = e2c_config.img_width
    shapes.append((h, w))
    for i in range(num_layers):
        h = conv_output_size(h, kernel_size=k[i], stride=s[i], padding=p[i])
        w = conv_output_size(w, kernel_size=k[i], stride=s[i], padding=p[i])
        shapes.append((h, w))
    return shapes

## MLP
class MLPReLU(nn.Module):
    """
    stack of linear+ReLU layers.
    """
    def __init__(self, in_dim, hidden_dims):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU(inplace=True))
            prev = h
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
    

def get_data_shape(data_dir: str = e2c_config.data_dir):
    frames_path = os.path.join(data_dir, e2c_config.frames_filename)
    actions_path = os.path.join(data_dir, e2c_config.actions_filename)

    # Use mmap_mode='r' so we don't load everything just to inspect the shape.
    frames = np.load(frames_path, mmap_mode="r")
    actions = np.load(actions_path, mmap_mode="r")

    # frames: (N, T, 1, H, W)
    _, T, _, H, W = frames.shape
    # actions: (N, T, A)
    _, _, u_dim = actions.shape

    x_dim = H * W

    return x_dim, u_dim, T