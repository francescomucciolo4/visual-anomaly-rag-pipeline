import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import numpy as np


class SSIMLoss(nn.Module):
    """
    SSIM Loss differenziabile completamente in PyTorch.
    Basato su: https://github.com/Po-Hsun-Su/pytorch-ssim
    """
    
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 3
        self.window = self.create_window(window_size, self.channel)
    
    def gaussian(self, window_size, sigma=1.5):
        gauss = torch.Tensor([
            np.exp(-(x - window_size//2)**2 / float(2*sigma**2)) 
            for x in range(window_size)
        ])
        return gauss / gauss.sum()
    
    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window
    
    def ssim(self, img1, img2, window, window_size, channel, size_average=True):
        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2
        
        C1 = 0.01**2
        C2 = 0.03**2
        
        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
        
        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)
    
    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        
        if self.window.device != img1.device:
            self.window = self.window.to(img1.device)
            self.window = self.window.type_as(img1)
        
        ssim_value = self.ssim(img1, img2, self.window, 
                               self.window_size, channel, self.size_average)
        
        return 1 - ssim_value

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.4, beta=0.6):
        super().__init__()
        self.alpha = alpha  # peso MSE
        self.beta = beta    # peso SSIM
        self.mse = nn.MSELoss()
        self.ssim = SSIMLoss()
    
    def forward(self, outputs, targets):
        mse_loss = self.mse(outputs, targets)
        ssim_loss = self.ssim(outputs, targets)
        return self.alpha * mse_loss + self.beta * ssim_loss


def train_autoencoder(model, train_loader, val_loader, device='cuda', 
                      num_epochs=50, lr=1e-3, use_ssim=True, patience=5):

    model = model.to(device)
    criterion = CombinedLoss(alpha=0.6, beta=0.4)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            batch = batch.to(device)
            outputs = model(batch)
            loss = criterion(outputs, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch.size(0)
        train_loss /= len(train_loader.dataset)
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                batch = batch.to(device)
                outputs = model(batch)
                loss = criterion(outputs, batch)
                val_loss += loss.item() * batch.size(0)
        val_loss /= len(val_loader.dataset)
        
        print(f"Epoch {epoch+1}: Train={train_loss:.4f}, Val={val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), "best_autoencoder.pth")
            print(f"  ✓ Best model saved (Val={best_val_loss:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                model.load_state_dict(torch.load("best_autoencoder.pth"))
                break
    
    return model


if __name__ == "__main__":
    from aecnn import CAE256_FC_Latent32
    from dataset_anomaly_detection import train_loader, val_loader

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = CAE256_FC_Latent32()
    
    model = train_autoencoder(
        model, train_loader, val_loader, 
        device=device, num_epochs=50, lr=1e-3, 
        use_ssim=True, patience=5
    )
    
