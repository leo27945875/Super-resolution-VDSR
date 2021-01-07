from PIL import Image

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm
from torchvision.transforms import Resize

from data_utils import normalize, denormalize


class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()
    
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class EDSR(nn.Module):
    def __init__(self, scale_factor, channels=256, n_residual=32, residual_scale=0.1):
        super(EDSR, self).__init__()
        self.scale = scale_factor
        self.block1   = nn.Conv2d(3, channels, kernel_size=3, padding=1)
        self.residual = nn.Sequential(*[ResidualBlock(channels, residual_scale) for _ in range(n_residual)])
        self.block2   = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.block3   = nn.Sequential(
            UpsampleBLock(channels, channels, scale_factor),
            nn.Conv2d(channels, 3, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x1 = self.block1(normalize(x))
        re = self.residual(x1) 
        x2 = self.block2(re)
        x3 = self.block3(x1 + x2)
        return denormalize(x3)


class WDSR(nn.Module):
    def __init__(self, scale_factor, channels=32, n_residual=64, residual_scale=0.1):
        super(WDSR, self).__init__()
        self.scale = scale_factor
        self.block1 = nn.Sequential(
            weight_norm(nn.Conv2d(3, channels, kernel_size=3, padding=1)),
            *[WDSR_B(channels, residual_scale) for _ in range(n_residual)],
            UpsampleBLock(channels, 3, scale_factor, kernel_size=5, is_wn=True)
        )
        self.block2 = UpsampleBLock(3, 3, scale_factor, kernel_size=5, is_wn=True)
    
    def forward(self, x):
        xn = normalize(x)
        x1 = self.block1(xn)
        x2 = self.block2(xn)
        x3 = x1 + x2
        return denormalize(x3)


class ResidualBlock(nn.Module):
    def __init__(self, channels, residual_scale=1., bn=False, activation=Mish):
        super(ResidualBlock, self).__init__()
        self.residual_scale = residual_scale
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.act = activation()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn = None
        
        if bn:
            self.bn = bn
            self.bn1 = nn.BatchNorm2d(channels, momentum=0.9)
            self.bn2 = nn.BatchNorm2d(channels, momentum=0.9)

    def forward(self, x):
        residual = self.conv1(x)
        if self.bn: residual = self.bn1(residual)
        residual = self.act(residual)
        residual = self.conv2(residual)
        if self.bn: residual = self.bn2(residual)
        return x + residual * self.residual_scale


class WDSR_B(nn.Module):
    def __init__(self, channels, residual_scale=1.):
        super(WDSR_B, self).__init__()
        self.residual_scale = residual_scale
        self.conv1 = weight_norm(nn.Conv2d(channels, channels * 6, kernel_size=1, padding=0))
        self.act = Mish()
        self.conv2 = weight_norm(nn.Conv2d(channels * 6, int(channels * 0.8), kernel_size=1, padding=0))
        self.conv3 = weight_norm(nn.Conv2d(int(channels * 0.8) , channels, kernel_size=3, padding=1))
    
    def forward(self, x):
        residual = self.conv1(x)
        residual = self.act(residual)
        residual = self.conv2(residual)
        residual = self.conv3(residual)
        return residual


class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, out_channels, up_scale, kernel_size=3, is_wn=False):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv2d(in_channels, 
                              out_channels * up_scale ** 2, 
                              kernel_size=kernel_size, 
                              padding=(kernel_size - 1) // 2)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        
        if is_wn: 
            self.conv = weight_norm(self.conv)
            
    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        return x


class SRResnet(nn.Module):
    def __init__(self, scale_factor):
        super(SRResnet, self).__init__()
        self.scale = scale_factor
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            Mish()
        )
        self.block2 = ResidualBlock(64, bn=True)
        self.block3 = ResidualBlock(64, bn=True)
        self.block4 = ResidualBlock(64, bn=True)
        self.block5 = ResidualBlock(64, bn=True)
        self.block6 = ResidualBlock(64, bn=True)
        self.block7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )
        self.block8 = nn.Sequential(
            UpsampleBLock(64, 64, scale_factor),
            nn.Conv2d(64, 3, kernel_size=3, padding=1)
        )

    def forward(self, x):
        block1 = self.block1(normalize(x))
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        block7 = self.block7(block6)
        block8 = self.block8(block1 + block7)

        return (torch.tanh(block8) + 1) / 2


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            Mish(),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64, momentum=0.9),
            Mish(),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128, momentum=0.9),
            Mish(),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128, momentum=0.9),
            Mish(),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256, momentum=0.9),
            Mish(),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256, momentum=0.9),
            Mish(),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512, momentum=0.9),
            Mish(),
            
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512, momentum=0.9),
            Mish(),

            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            
            nn.Linear(512, 1024),
            Mish(),
            nn.Linear(1024, 1)
        )
        
        self.initialize()

    def forward(self, x):
        return self.net(x)
    
    def initialize(self):
        for m in self.net.modules():
            if type(m) is nn.Conv2d:
                torch.nn.init.normal_(m.weight, std=0.02)
            
            if type(m) is nn.BatchNorm2d:
                torch.nn.init.normal_(m.weight, mean=1., std=0.02)
    
    
    
    
