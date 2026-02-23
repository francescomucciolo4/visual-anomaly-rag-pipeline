import torch
import torch.nn as nn

class CAE256_Latent100(nn.Module):
    def __init__(self):
        super(CAE256_Latent100, self).__init__()
        
        # ---- Encoder convoluzionale ----
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),    # 256x256 -> 128x128
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),   # 128x128 -> 64x64
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),   # 64x64 -> 32x32
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 32x32 -> 16x16
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), # 16x16 -> 4x4
            nn.ReLU()
        )
        
       # Flatten per passare a Linear
        self.flatten = nn.Flatten()
        self.fc_enc = nn.Linear(256*8*8, 100)  # 16384 -> 100

        # ---- Decoder ----
        self.fc_dec = nn.Linear(100, 256*8*8)  # 100 -> 16384
        self.unflatten = nn.Unflatten(1, (256, 8, 8))  # ritorna a [256, 8, 8]
        
        self.decoder_conv = nn.Sequential(
        nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # 8 → 16
        nn.ReLU(),

        nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),   # 16 → 32
        nn.ReLU(),

        nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),    # 32 → 64
        nn.ReLU(),

        nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),    # 64 → 128
        nn.ReLU(),

        nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),     # 128 → 256
        nn.Sigmoid()
    )
        
    def forward(self, x):
        # Encoder
        x = self.encoder_conv(x)
        x = self.flatten(x)
        latent = self.fc_enc(x)  # vettore latente di dimensione 100
        
        # Decoder
        x = self.fc_dec(latent)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        return x
