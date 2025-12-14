import torch
from torch import nn
import numpy as np

from models.dynamics.rnn_dynamics import RecurrentTransition
from models.vae.conv_vae import ConvVAE
from models.e2c.common import (
    E2COutput,
    recons_loss,
    kl_diag_gaussian,
    sample_normal,
    mse_loss,
)
from config import e2c_config


class RecurrentE2C(nn.Module):
    """
    GRU for dynamics, Conv VAE
    """
    def __init__(self, x_dim, u_dim):
        super().__init__()        
        self.vae = ConvVAE()
        self.transition = RecurrentTransition(u_dim, z_dim=self.vae.latent_dim)

        self.u_dim = u_dim
        self.x_dim = x_dim

    def forward(self, x_seq, u_seq):
        device = x_seq.device
        B, Kp1, C, H, W = x_seq.size()
        K = Kp1 - 1

        # init RNN
        h_t = self.transition.init_state(B, device)

        L_x_list = []
        L_x_tp1_list = []
        KL_list = []
        mse_list = []
        mse_tp1_list = []

        for t in range(K):
            # contiguous for torch.compile
            x_t   = x_seq[:, t].contiguous()
            x_tp1 = x_seq[:, t + 1].contiguous() 
            u_t   = u_seq[:, t, :]

            mu_z_t, logvar_z_t = self.vae.encoder(x_t)
            z_t = self.vae.reparameterize(mu_z_t, logvar_z_t)

            x_recons_t, _ = self.vae.decoder(z_t)

            L_x_t = recons_loss(x_t.view(B, -1), x_recons_t.view(B, -1))
            mse_x_t = mse_loss(x_t.view(B, -1), x_recons_t.view(B, -1))

            mu_z_t_hat, logvar_z_t_hat = self.transition.prior(h_t)
            KL_t = kl_diag_gaussian(mu_z_t, logvar_z_t, mu_z_t_hat, logvar_z_t_hat)
            h_tp1 = self.transition.gru_step(z_t, u_t, h_t)

            mu_z_tp1_hat, logvar_z_tp1_hat = self.transition.prior(h_tp1)
            z_prior_tp1 = sample_normal(mu_z_tp1_hat, torch.exp(0.5 * logvar_z_tp1_hat))
            x_pred_tp1, _ = self.vae.decoder(z_prior_tp1)
            L_x_tp1 = recons_loss(x_tp1.view(B, -1), x_pred_tp1.view(B, -1))
            mse_tp1 = mse_loss(x_tp1.view(B, -1), x_pred_tp1.view(B, -1))

            L_x_list.append(L_x_t)
            L_x_tp1_list.append(L_x_tp1)
            KL_list.append(KL_t)
            mse_list.append(mse_x_t)
            mse_tp1_list.append(mse_tp1)

            h_t = h_tp1
        L_x=torch.stack(L_x_list, dim=1).mean(dim=1)
        L_x_tp1=torch.stack(L_x_tp1_list, dim=1).mean(dim=1)
        KL=torch.stack(KL_list, dim=1).mean(dim=1)
        mse_x=torch.stack(mse_list, dim=1).mean(dim=1)
        mse_x_tp1=torch.stack(mse_tp1_list, dim=1).mean(dim=1)

        loss = (L_x + L_x_tp1 + e2c_config.beta_kl * KL).mean()

        return E2COutput(
            loss=loss,
            L_x=L_x.mean(),
            L_x_tp1=L_x_tp1.mean(),
            KL=KL.mean(),
            mse_x=mse_x.mean(),
            mse_x_tp1=mse_x_tp1.mean(),
        )
