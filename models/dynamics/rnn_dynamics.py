import torch
from torch import nn
from config import e2c_config

class RecurrentTransition(nn.Module):
    def __init__(self, u_dim, z_dim: int | None = None):
        super().__init__()

        self.h_dim = e2c_config.rnn_hidden_dim
        self.z_dim = z_dim if z_dim is not None else e2c_config.latent_dim
        self.u_dim = u_dim

        self.gru = nn.GRUCell(self.z_dim + self.u_dim, self.h_dim)
        self.prior_head = nn.Linear(self.h_dim, 2 * self.z_dim)

    def init_state(self, batch_size, device):
        return torch.zeros(batch_size, self.h_dim, device=device)

    def prior(self, h_t):
        stats = self.prior_head(h_t)
        mu, logvar = torch.chunk(stats, 2, dim=1)
        return mu, logvar

    def gru_step(self, z_t, u_t, h_t):
        inp = torch.cat([z_t, u_t], dim=1)   # (batch, z_dim + u_dim)
        h_tp1 = self.gru(inp, h_t)
        return h_tp1