import torch.nn as nn
import networks.blocks as B

class AutoEncoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            B.ConvBlock(mode='C', in_channels=3, out_channels=64, kernel_size=3,  act_type='relu')
            nn.ReLU(True),
            B.ConvBlock(mode='C', in_channels=64, out_channels=64, kernel_size=3,  act_type='relu')
            nn.ReLU(True))
            # nn.AdaptiveAvgPool2d(1)
        self.decoder = nn.Sequential(
            B.ConvBlock(mode='C', in_channels=64, out_channels=64, kernel_size=3,  act_type='relu')
            nn.ReLU(True),
            nn.Linear(256, 28 * 28),
            nn.Sigmoid())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x