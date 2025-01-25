import torch
import torch.nn as nn
import torch.nn.functional as F

class DenoisNet(nn.Module):
    def __init__(self, input_channels):
        super(DenoisNet, self).__init__()
        
        # Encoder
        self.encoder_conv1 = nn.Conv1d(input_channels, 64, kernel_size=3, padding='same')
        self.encoder_pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.encoder_conv2 = nn.Conv1d(64, 128, kernel_size=3, padding='same')
        self.encoder_pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.encoder_conv3 = nn.Conv1d(128, 256, kernel_size=3, padding='same')
        self.encoder_pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Bottleneck
        self.bottleneck_conv = nn.Conv1d(256, 512, kernel_size=3, padding='same')
        
        # Decoder
        self.decoder_up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.decoder_conv1 = nn.Conv1d(512 + 256, 256, kernel_size=3, padding='same')  # Adjust input channels
        
        self.decoder_up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.decoder_conv2 = nn.Conv1d(256 + 128, 128, kernel_size=3, padding='same')  # Adjust input channels
        
        self.decoder_up3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.decoder_conv3 = nn.Conv1d(128 + 64, 64, kernel_size=3, padding='same')  # Adjust input channels
        
        # Output
        self.output_conv = nn.Conv1d(64, input_channels, kernel_size=3, padding='same')
    
    def forward(self, x):
        # Encoder
        x1 = F.relu(self.encoder_conv1(x))
        x1_pool = self.encoder_pool1(x1)
        
        x2 = F.relu(self.encoder_conv2(x1_pool))
        x2_pool = self.encoder_pool2(x2)
        
        x3 = F.relu(self.encoder_conv3(x2_pool))
        x3_pool = self.encoder_pool3(x3)
        
        # Bottleneck
        x4 = F.relu(self.bottleneck_conv(x3_pool))
        
        # Decoder
        x5 = self.decoder_up1(x4)
        x5 = torch.cat([x5, x3], dim=1)  # Skip connection (512 + 256 channels)
        x5 = F.relu(self.decoder_conv1(x5))
        
        x6 = self.decoder_up2(x5)
        x6 = torch.cat([x6, x2], dim=1)  # Skip connection (256 + 128 channels)
        x6 = F.relu(self.decoder_conv2(x6))
        
        x7 = self.decoder_up3(x6)
        x7 = torch.cat([x7, x1], dim=1)  # Skip connection (128 + 64 channels)
        x7 = F.relu(self.decoder_conv3(x7))
        
        # Output
        output = torch.sigmoid(self.output_conv(x7))
        return output