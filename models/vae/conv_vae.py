import torch
from torch import nn
from einops import rearrange

from config import e2c_config

from utils import get_vae_hidden_dims, get_vae_encoder_shapes, MLPReLU

class ConvEncoder(nn.Module):
    """
    Given: N images concatenated along channel x
    Returns: latent diagonal Gaussian parameters mu_z, logvar_z
    """
    def __init__(self, in_channels: int, latent_dim: int):
        super().__init__()

        # Read CNN spec from config
        conv_channels = getattr(e2c_config, "conv_channels", [32, 64, 128])
        num_layers = len(conv_channels)
        k = getattr(e2c_config, "conv_kernel_size", 4)
        s = getattr(e2c_config, "conv_stride", 2)
        p = getattr(e2c_config, "conv_padding", 1)

        # Broadcast scalar hyperparams to per-layer lists if needed
        if not isinstance(k, (list, tuple)):
            k = [k] * num_layers
        if not isinstance(s, (list, tuple)):
            s = [s] * num_layers
        if not isinstance(p, (list, tuple)):
            p = [p] * num_layers

        layers = []
        prev_c = in_channels
        for i in range(num_layers):
            layers.append(nn.Conv2d(prev_c, conv_channels[i], k[i], s[i], p[i]))
            layers.append(nn.ReLU(inplace=True))
            prev_c = conv_channels[i]
        self.cnn = nn.Sequential(*layers)

        _, _, flatten_dim = get_vae_hidden_dims()

        self.mlp = MLPReLU(flatten_dim, hidden_dims=[150, 150, 150, latent_dim*2])

    def forward(self, x):
        h = self.cnn(x)
        h_view = rearrange(h, "... c h w -> ... (c h w)")
        stats = self.mlp(h_view)
        mu_z, logvar_z = torch.chunk(stats, 2, dim=-1)
        return mu_z, logvar_z

class ConvDecoder(nn.Module):
    """
    Given: latent variable z
    Returns: reconstructed image \hat{x}
    """
    def __init__(self, out_channels: int, latent_dim: int):
        super().__init__()

        conv_channels = getattr(e2c_config, "conv_channels", [32, 64, 128])
        num_layers = len(conv_channels)
        k = getattr(e2c_config, "conv_kernel_size", 4)
        s = getattr(e2c_config, "conv_stride", 2)
        p = getattr(e2c_config, "conv_padding", 1)

        if not isinstance(k, (list, tuple)):
            k = [k] * num_layers
        if not isinstance(s, (list, tuple)):
            s = [s] * num_layers
        if not isinstance(p, (list, tuple)):
            p = [p] * num_layers

        self.h4, self.w4, flatten_dim = get_vae_hidden_dims()
        last_c = conv_channels[-1]
        self.fc_z = nn.Linear(latent_dim, last_c * self.h4 * self.w4)

        deconv_layers = []
        rev_channels = list(conv_channels)[::-1]
        rev_k = list(k)[::-1]
        rev_s = list(s)[::-1]
        rev_p = list(p)[::-1]

        enc_shapes = get_vae_encoder_shapes()
        dec_target_shapes = list(enc_shapes)[-2::-1]
        curr_h, curr_w = enc_shapes[-1]

        prev_c = rev_channels[0]
        for i in range(1, len(rev_channels)):
            target_h, target_w = dec_target_shapes[i-1]
            base_h = (curr_h - 1) * rev_s[i-1] - 2 * rev_p[i-1] + rev_k[i-1]
            base_w = (curr_w - 1) * rev_s[i-1] - 2 * rev_p[i-1] + rev_k[i-1]
            out_pad_h = max(0, min(rev_s[i-1] - 1, target_h - base_h))
            out_pad_w = max(0, min(rev_s[i-1] - 1, target_w - base_w))
            deconv_layers.append(
                nn.ConvTranspose2d(
                    prev_c, rev_channels[i], rev_k[i-1], rev_s[i-1], rev_p[i-1],
                    output_padding=(out_pad_h, out_pad_w)
                )
            )
            deconv_layers.append(nn.ReLU(inplace=True))
            prev_c = rev_channels[i]
            curr_h, curr_w = target_h, target_w

        target_h, target_w = enc_shapes[0]
        base_h = (curr_h - 1) * rev_s[-1] - 2 * rev_p[-1] + rev_k[-1]
        base_w = (curr_w - 1) * rev_s[-1] - 2 * rev_p[-1] + rev_k[-1]
        out_pad_h = max(0, min(rev_s[-1] - 1, target_h - base_h))
        out_pad_w = max(0, min(rev_s[-1] - 1, target_w - base_w))
        deconv_layers.append(
            nn.ConvTranspose2d(
                prev_c, out_channels, rev_k[-1], rev_s[-1], rev_p[-1],
                output_padding=(out_pad_h, out_pad_w)
            )
        )

        self.dcnn = nn.Sequential(*deconv_layers)

    def forward(self, z):
        h = self.fc_z(z)
        conv_channels = getattr(e2c_config, "conv_channels", [32, 64, 128])
        last_c = conv_channels[-1]
        h_view = torch.reshape(h, (z.size(0), last_c, self.h4, self.w4))
        logits = self.dcnn(h_view)
        x_hat = torch.sigmoid(logits)
        return x_hat, h_view

class VAE(nn.Module):
    def __init__(self, in_channels=None):
        super().__init__()

        if in_channels is None:
            self.in_channels = e2c_config.state_seq_len * e2c_config.channels_per_img
        else:
            self.in_channels = in_channels
            
        self.latent_dim = e2c_config.latent_dim

        self.encoder = ConvEncoder(self.in_channels, self.latent_dim)
        out_channels = self.in_channels
        self.decoder = ConvDecoder(out_channels, self.latent_dim)
    
    @staticmethod
    def reparameterize(mu_z, logvar_z):
        epsilon = torch.randn_like(logvar_z)
        return mu_z + epsilon * 0.5 * logvar_z.exp()

    def forward(self, x):
        mu_z, logvar_z = self.encoder(x)
        z = self.reparameterize(mu_z, logvar_z)
        x_hat = self.decoder(z)
        return {
            "mu_z": mu_z,
            "logvar_z": logvar_z,
            "z": z,
            "x_hat": x_hat
        }
    

ConvVAE = VAE

if __name__ == "__main__":
    # quick test
    x = torch.zeros(2, 1, e2c_config.img_height, e2c_config.img_width)
    vae = VAE()
    res = vae(x)
    x_hat = res["x_hat"]
    assert x.shape == x_hat.shape, f"x.shape {x.shape}, x_hat.shape {x_hat.shape}"
    print(x.shape)
