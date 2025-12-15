import torch
import torch.nn as nn
from .attention import CBAM
from compressai.entropy_models import EntropyBottleneck, GaussianConditional

class MotionCompressor(nn.Module):
    """
    Simple Autoencoder for Motion Vector Compression
    """
    def __init__(self, in_channels=2, out_channels=2):
        super(MotionCompressor, self).__init__()
        self.entropy_bottleneck = EntropyBottleneck(128)
        
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, stride=2, padding=1),
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 128, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 128, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, out_channels, 3, stride=2, padding=1, output_padding=1),
        )

    def forward(self, x):
        y = self.encoder(x)
        y_hat, y_likelihoods = self.entropy_bottleneck(y)
        x_hat = self.decoder(y_hat)
        return x_hat, y_likelihoods

class ResidualCompressor(nn.Module):
    """
    Residual Compression Network with optional CBAM
    """
    def __init__(self, in_channels=3, out_channels=3, use_attention=True):
        super(ResidualCompressor, self).__init__()
        self.use_attention = use_attention
        self.entropy_bottleneck = EntropyBottleneck(192)
        
        # Encoder
        self.conv1 = nn.Conv2d(in_channels, 128, 5, stride=2, padding=2)
        self.cbam1 = CBAM(128) if use_attention else nn.Identity()
        
        self.conv2 = nn.Conv2d(128, 128, 5, stride=2, padding=2)
        self.cbam2 = CBAM(128) if use_attention else nn.Identity()
        
        self.conv3 = nn.Conv2d(128, 128, 5, stride=2, padding=2)
        self.cbam3 = CBAM(128) if use_attention else nn.Identity()
        
        self.conv4 = nn.Conv2d(128, 192, 5, stride=2, padding=2)
        
        # Decoder
        self.deconv1 = nn.ConvTranspose2d(192, 128, 5, stride=2, padding=2, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(128, 128, 5, stride=2, padding=2, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(128, 128, 5, stride=2, padding=2, output_padding=1)
        self.deconv4 = nn.ConvTranspose2d(128, out_channels, 5, stride=2, padding=2, output_padding=1)
        
        self.relu = nn.ReLU()

    def forward(self, x):
        # Encoder
        out = self.conv1(x)
        if self.use_attention: out = self.cbam1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        if self.use_attention: out = self.cbam2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        if self.use_attention: out = self.cbam3(out)
        out = self.relu(out)
        
        y = self.conv4(out)
        
        # Entropy Bottleneck
        y_hat, y_likelihoods = self.entropy_bottleneck(y)
        
        # Decoder
        out = self.relu(self.deconv1(y_hat))
        out = self.relu(self.deconv2(out))
        out = self.relu(self.deconv3(out))
        x_hat = self.deconv4(out)
        
        return x_hat, y_likelihoods
