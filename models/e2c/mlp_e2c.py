import torch
from torch import nn
import numpy as np

from models.dynamics.mlp_dynamics import MLPTransition
from models.vae.conv_vae import ConvVAE
from models.e2c.common import (
    E2COutput,
    recons_loss,
    kl_diag_gaussian,
    mse_loss,
)
from config import e2c_config


class MLP_E2C(nn.Module):
    """
    MLP dynamics, Convolutional VAE 
    """
    def __init__(self, x_dim, u_dim):
        super().__init__()        
        self.vae = ConvVAE()
        self.transition = MLPTransition(e2c_config.latent_dim, u_dim)

        self.u_dim = u_dim
        self.x_dim = x_dim

    @staticmethod
    def kl_gaussian_e2c(mu_q, logsigma_q, sigma_q, v, r, mu_p, logsigma_p, sigma_p, eps: float):
        """
        see appendix A1 of embed to control for the covariance parameterization as
        A = I + outer(v,r)
        covariance of Q = A diag(sigma_q^2) A^T
        """
        sum_feat = lambda x: x.sum(dim=1)
        k = mu_q.size(1)
        s02 = sigma_q.pow(2)
        s12 = sigma_p.pow(2) + eps

        trace = sum_feat(s02 * (1.0 + 2.0 * v * r) / s12) \
            + sum_feat(v.pow(2) / s12) * sum_feat(r.pow(2) * s02)
        mean_diff = sum_feat((mu_p - mu_q).pow(2) / s12)
        det = 2.0 * (sum_feat(logsigma_p - logsigma_q) - torch.log(1.0 + sum_feat(v * r)))

        return 0.5 * (trace + mean_diff - k + det)

    def forward(self, x_seq, u_seq):
        """
        images stacked along channel (k)
        controls between each pair of images (k-1)
        """
        B, Kp1, C, H, W = x_seq.size()
        K = Kp1 - 1

        L_x_list = []
        L_x_tp1_list = []
        KL_prior_list = []
        KL_trans_list = []
        mse_list = []
        mse_tp1_list = []

        lambd = getattr(e2c_config, "lambda_transition_kl", 0.25)
        eps = getattr(e2c_config, "eps", 1e-9)

        for t in range(K):
            # torch.compile needs contiguous data
            x_t   = x_seq[:, t].contiguous()
            x_tp1 = x_seq[:, t + 1].contiguous()
            u_t   = u_seq[:, t, :]

            mu_z_t, logvar_z_t = self.vae.encoder(x_t)
            z_t = self.vae.reparameterize(mu_z_t, logvar_z_t)
                                          
            x_recons_tuple = self.vae.decoder(z_t)
            x_recons_t = x_recons_tuple[0]

            L_x_t = recons_loss(x_t.reshape(B, -1), x_recons_t.reshape(B, -1))
            mse_x_t = mse_loss(x_t.reshape(B, -1), x_recons_t.reshape(B, -1))

            # KL for standard diagonal gaussian (log of one is zero)
            KL_prior_t = kl_diag_gaussian(
                mu_z_t, logvar_z_t, torch.zeros_like(mu_z_t), torch.zeros_like(logvar_z_t)
            )

            A_t, B_t, o_t, v_t, r_t = self.transition(z_t)
            z_tp1_hat = self.transition.forward_dynamics(A_t, B_t, o_t, z_t, u_t)
            x_pred_tp1, h_view = self.vae.decoder(z_tp1_hat)

            L_x_tp1 = recons_loss(x_tp1.reshape(B, -1), x_pred_tp1.reshape(B, -1))  # (B,)
            mse_tp1 = mse_loss(x_tp1.reshape(B, -1), x_pred_tp1.reshape(B, -1))


            sigma_t = torch.exp(0.5 * logvar_z_t)
            logsigma_t = 0.5 * logvar_z_t

            mu_tp1 = torch.einsum("bij,bj->bi", A_t, mu_z_t) \
                      + torch.einsum("bdu,bu->bd", B_t, u_t) + o_t

            mu_z_tp1, logvar_z_tp1 = self.vae.encoder(x_tp1)
            sigma_tp1 = torch.exp(0.5 * logvar_z_tp1)
            logsigma_tp1 = 0.5 * logvar_z_tp1

            KL_trans_t = self.kl_gaussian_e2c(
                mu_q=mu_tp1,
                logsigma_q=logsigma_t,
                sigma_q=sigma_t,
                v=v_t,
                r=r_t,
                mu_p=mu_z_tp1,
                logsigma_p=logsigma_tp1,
                sigma_p=sigma_tp1,
                eps=eps,
            )

            L_x_list.append(L_x_t)
            L_x_tp1_list.append(L_x_tp1)
            KL_prior_list.append(KL_prior_t)
            KL_trans_list.append(KL_trans_t)
            mse_list.append(mse_x_t)
            mse_tp1_list.append(mse_tp1)

        L_x = torch.stack(L_x_list, dim=1).mean(dim=1)
        L_x_tp1 = torch.stack(L_x_tp1_list, dim=1).mean(dim=1)
        KL_prior = torch.stack(KL_prior_list, dim=1).mean(dim=1)
        KL_trans = torch.stack(KL_trans_list, dim=1).mean(dim=1)
        mse_x = torch.stack(mse_list, dim=1).mean(dim=1)
        mse_x_tp1 = torch.stack(mse_tp1_list, dim=1).mean(dim=1)

        # kl_beta is usually 1.0 but keep for tuning
        loss = (L_x + L_x_tp1 + e2c_config.beta_kl * KL_prior + lambd * KL_trans).mean()

        return E2COutput(
            loss=loss,
            L_x=L_x.mean(),
            L_x_tp1=L_x_tp1.mean(),
            KL=KL_trans.mean(),
            mse_x=mse_x.mean(),
            mse_x_tp1=mse_x_tp1.mean(),
        )
