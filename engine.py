import torch
from config import e2c_config
from tqdm.auto import tqdm

def train_epoch(model, train_loader, optimizer, device, writer, global_step):
    model.train()
    total_loss = 0
    total_kl = 0
    total_l_x = 0
    total_l_x_tp1 = 0
    total_mse_x = 0
    total_mse_x_tp1 = 0
    num_batches = 0
    
    for x_seq, u_seq in tqdm(train_loader, desc="Training"):
        x_seq = x_seq.to(device)  # (B, K+1, stacked_frames, H, W)
        u_seq = u_seq.to(device)  # (B, K, u_dim)
        
        optimizer.zero_grad()
        out = model(x_seq, u_seq)
        loss = out.loss
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_kl += out.KL.item()
        total_l_x += out.L_x.item()
        total_l_x_tp1 += out.L_x_tp1.item()
        total_mse_x += out.mse_x.item()
        total_mse_x_tp1 += out.mse_x_tp1.item()
        num_batches += 1
        
        # tensorboard
        if global_step % e2c_config.iterations_per_log == 0:
            writer.add_scalars("loss/train", {
                "total": loss.item(),
                "kl": out.KL.item(),
                "l_x": out.L_x.item(),
                "l_x_tp1": out.L_x_tp1.item(),
                "mse_x": out.mse_x.item(),
                "mse_x_tp1": out.mse_x_tp1.item(),
            }, global_step)

            writer.add_scalars("mse/train", {
                "recon": out.mse_x.item(),
                "next": out.mse_x_tp1.item(),
            }, global_step)
        
        global_step += 1
    
    avg_loss = total_loss / num_batches
    avg_kl = total_kl / num_batches
    avg_l_x = total_l_x / num_batches
    avg_l_x_tp1 = total_l_x_tp1 / num_batches
    avg_mse_x = total_mse_x / num_batches
    avg_mse_x_tp1 = total_mse_x_tp1 / num_batches
    
    return avg_loss, avg_kl, avg_l_x, avg_l_x_tp1, avg_mse_x, avg_mse_x_tp1, global_step


def validate(model, val_loader, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    total_kl = 0
    total_l_x = 0
    total_l_x_tp1 = 0
    total_mse_x = 0
    total_mse_x_tp1 = 0
    num_batches = 0
    
    with torch.inference_mode():
        for x_seq, u_seq in val_loader:
            x_seq = x_seq.to(device)
            u_seq = u_seq.to(device)
            
            out = model(x_seq, u_seq)
            
            total_loss += out.loss.item()
            total_kl += out.KL.item()
            total_l_x += out.L_x.item()
            total_l_x_tp1 += out.L_x_tp1.item()
            total_mse_x += out.mse_x.item()
            total_mse_x_tp1 += out.mse_x_tp1.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_kl = total_kl / num_batches
    avg_l_x = total_l_x / num_batches
    avg_l_x_tp1 = total_l_x_tp1 / num_batches
    avg_mse_x = total_mse_x / num_batches
    avg_mse_x_tp1 = total_mse_x_tp1 / num_batches
    
    return avg_loss, avg_kl, avg_l_x, avg_l_x_tp1, avg_mse_x, avg_mse_x_tp1
