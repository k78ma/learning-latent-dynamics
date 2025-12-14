import torch
from typing import NamedTuple


eps = 1e-9  # numerical stability


class E2COutput(NamedTuple):
    """because Tensorboard wants NamedTuples"""
    loss: torch.Tensor
    L_x: torch.Tensor
    L_x_tp1: torch.Tensor
    KL: torch.Tensor
    mse_x: torch.Tensor
    mse_x_tp1: torch.Tensor


def sample_normal(mu, sigma):
    """
    reparameterization to separate noise and statistical variables
    """
    eps_ = torch.randn_like(sigma)
    return mu + sigma * eps_


def binary_crossentropy_ll(target, output):
    """
    log likelihood BCE
    """
    return target * torch.log(output + eps) + \
           (1.0 - target) * torch.log(1.0 - output + eps)


def recons_loss(x, x_recons):
    ll = binary_crossentropy_ll(x, x_recons).sum(dim=1)
    return -ll


def mse_loss(x, x_recons):
    return torch.mean((x - x_recons) ** 2, dim=1)


def kl_diag_gaussian(mu_q, logvar_q, mu_p, logvar_p):
    """
    KL divergence for two diagonal Gaussians
    """
    var_q = torch.exp(logvar_q)
    var_p = torch.exp(logvar_p)
    
    # per-dim KL, then sum over latent dim
    kl = var_q / (var_p + eps) + (mu_q - mu_p) ** 2 / (var_p + eps) \
         - 1.0 + (logvar_p - logvar_q)
    return 0.5 * kl.sum(dim=1)
