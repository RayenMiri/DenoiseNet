import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )
    
    def forward(self, x):
        return self.block(x)

class DenoiseNet(nn.Module):
    def __init__(self, n_fft=512, hop_length=128):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        # Encoder
        self.enc1 = UNetBlock(2, 32)
        self.enc2 = UNetBlock(32, 64)
        self.enc3 = UNetBlock(64, 128)
        
        # Decoder
        self.dec1 = UNetBlock(128 + 64, 64)
        self.dec2 = UNetBlock(64 + 32, 32)
        self.dec3 = UNetBlock(32, 2)
        
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        # x: [batch, 2, freq, time] (magnitude + phase)
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool(e1)
        e2 = self.enc2(p1)
        p2 = self.pool(e2)
        e3 = self.enc3(p2)
        
        # Decoder with dimension alignment
        u1 = self.upsample(e3)
        u1 = self.match_tensor_size(u1, e2)  # Align dimensions
        d1 = self.dec1(torch.cat([u1, e2], dim=1))
        u2 = self.upsample(d1)
        u2 = self.match_tensor_size(u2, e1)  # Align dimensions
        d2 = self.dec2(torch.cat([u2, e1], dim=1))
        out = self.dec3(d2)
        
        return out

    def match_tensor_size(self, tensor, target_tensor):
        """
        Adjust `tensor` to match the spatial dimensions of `target_tensor`.
        If `tensor` is larger, crop it. If smaller, pad it.
        """
        _, _, h, w = tensor.size()
        _, _, th, tw = target_tensor.size()
        
        # Crop if tensor is larger
        if h > th:
            tensor = tensor[:, :, :th, :]
        if w > tw:
            tensor = tensor[:, :, :, :tw]
        
        # Pad if tensor is smaller
        if h < th or w < tw:
            pad_h = th - h
            pad_w = tw - w
            tensor = F.pad(tensor, (0, pad_w, 0, pad_h))  # Pad width and height
        
        return tensor
