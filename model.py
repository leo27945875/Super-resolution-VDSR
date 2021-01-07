import torch
import torch.nn as nn
import torch.nn.functional as F


class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()
    
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class ConvReLUBlock(nn.Module):
    def __init__(self, midChannels):
        super().__init__()
        self.conv   = nn.Conv2d(in_channels=midChannels, out_channels=midChannels, kernel_size=3, stride=1, padding=1, bias=False)
        self.active = Mish()
        
    def forward(self, x):
        return self.active(self.conv(x)) + x


class VDSR(nn.Module):
    def __init__(self, numConv=24, inChannels=1, midChannels=64, isInit=True):
        super().__init__()
        self.residual = self.MakeLayers(numConv - 2, midChannels)
        self.input    = nn.Conv2d(in_channels=inChannels, out_channels=midChannels, kernel_size=3, stride=1, padding=1, bias=False)
        self.output   = nn.Conv2d(in_channels=midChannels, out_channels=inChannels, kernel_size=3, stride=1, padding=1, bias=False)
        self.active   = Mish()
        self.Initialize(isInit)
                
    def MakeLayers(self, num, midChannels):
        layers = []
        for _ in range(num):
            layers.append(ConvReLUBlock(midChannels=midChannels))

        return nn.Sequential(*layers)
    
    def Initialize(self, isInit):
        if isInit:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        out = self.active(self.input(x))
        out = self.residual(out)
        out = self.output(out)
        out = torch.add(out, x)
        return out


