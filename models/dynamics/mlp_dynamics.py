import torch
from torch import nn
from utils import MLPReLU
from config import e2c_config

class MLPTransition(nn.Module):
    """
    Transition network: given current latent z and control u, approximate local linearization:
    A, B, o with A = I + v*r^T
    """
    def __init__(self, z_dim, u_dim):
        super().__init__()
        hidden_dims = getattr(e2c_config, "mlp_hidden_dims", [100, 100])
        assert len(hidden_dims) >= 1, "mlp_hidden_dims must have at least one layer"
        self.mlp = MLPReLU(z_dim, hidden_dims)
        last_h = hidden_dims[-1]
        self.A_head = nn.Linear(last_h, 2 * z_dim) # v, r
        self.B_head = nn.Linear(last_h, z_dim * u_dim)
        self.o_head = nn.Linear(last_h, z_dim)
        self.z_dim = z_dim
        self.u_dim = u_dim

    def forward(self, z):
        h = self.mlp(z)

        # latent state transition matrix 
        vr = self.A_head(h)
        v, r = torch.chunk(vr, 2, dim=1) 
        v1 = v.unsqueeze(2)
        rT = r.unsqueeze(1)
        outer = torch.bmm(v1, rT)

        batch_size = z.size(0)
        identity = torch.eye(self.z_dim, device=z.device).unsqueeze(0).expand(batch_size, -1, -1)
        A = identity + outer

        # control matrix
        B = self.B_head(h)
        B = B.view(batch_size, self.z_dim, self.u_dim)

        # offset
        o = self.o_head(h)

        return A, B, o, v, r
    
    @staticmethod
    def forward_dynamics(A, B, o, z, u):
        z_hat = torch.einsum("...dm,...m->...d", A, z) \
        + torch.einsum("...du,...u->...d", B, u) + o
        return z_hat