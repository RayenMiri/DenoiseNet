import torch
import torch.nn as nn

class DenoiseUNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Encoder (Pooling applied only to the time dimension)
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=(1, 1)),  # (freq, time)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))  # Pool time dimension
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))  # Pool time dimension
        )
        
        # Decoder (Upsampling applied only to the time dimension)
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=(1, 2), stride=(1, 2)),  # Upsample time
            nn.ReLU()
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(32, 1, kernel_size=(1, 2), stride=(1, 2)),  # Upsample time
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Input shape: [B, 1, freq=257, time]
        x1 = self.enc1(x)  # [B, 32, 257, time/2]
        x2 = self.enc2(x1)  # [B, 64, 257, time/4]
        
        x = self.dec1(x2)  # [B, 32, 257, time/2]
        x = self.dec2(x)  # [B, 1, 257, time]
        return x