import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class ResBlock(nn.Module):
    def __init__(self, channel=64, Dilation=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=Dilation, dilation=Dilation)
        self.conv2 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=Dilation, dilation=Dilation)

    def forward(self, x):
        out = self.conv1(F.selu(x, False))
        out = self.conv2(F.selu(out, True))

        out += x
        return out


class Tunnel(nn.Module):
    def __init__(self, len=1):
        super(Tunnel, self).__init__()

        tunnel = [ResBlock() for _ in range(len)]
        self.tunnel = nn.Sequential(*tunnel)

    def forward(self, x):
        return self.tunnel(x)


class DResBlock(nn.Module):
    def __init__(self, channel=64, Dilation=1):
        super(DResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=Dilation, dilation=Dilation)
        self.conv2 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=Dilation, dilation=Dilation)

    def forward(self, x):
        out = self.conv1(F.selu(x, False))
        out = self.conv2(F.selu(out, True))

        out += x
        return out


class DTunnel(nn.Module):
    def __init__(self, len=1, channel=64):
        super(DTunnel, self).__init__()

        tunnel = [DResBlock(channel) for _ in range(len)]
        self.tunnel = nn.Sequential(*tunnel)

    def forward(self, x):
        return self.tunnel(x)


class Gnet(nn.Module):
    def __init__(self):
        super(Gnet, self).__init__()

        self.model = nn.Sequential(nn.Conv2d(100, 1024, 1, 1, bias=False),
                                   nn.PixelShuffle(4),
                                   nn.SELU(inplace=True),
                                   nn.Conv2d(64, 256, 3, 1, 1, bias=False),
                                   nn.PixelShuffle(2),
                                   nn.Conv2d(64, 256, 3, 1, 1, bias=False),
                                   nn.PixelShuffle(2),
                                   Tunnel(16),
                                   nn.Conv2d(64, 256, 3, 1, 1, bias=False),
                                   nn.PixelShuffle(2),
                                   nn.SELU(inplace=True),
                                   nn.Conv2d(64, 256, 3, 1, 1, bias=False),
                                   nn.PixelShuffle(2),
                                   nn.SELU(inplace=True),
                                   nn.Conv2d(64, 3, 3, 1, 1, bias=False),
                                   nn.Tanh()
                                   )

    def forward(self, x):
        return self.model(x)


class PatchD(nn.Module):
    def __init__(self, ndf=64):
        super(PatchD, self).__init__()

        sequence = [
            nn.Conv2d(3, ndf, kernel_size=4, stride=2, padding=1),
            nn.SELU(True),
            DTunnel(2, ndf),

        ]  # 32

        sequence += [
            nn.Conv2d(ndf * 1, ndf * 2,
                      kernel_size=4, stride=2, padding=1),
            nn.SELU(True),
            DTunnel(2, ndf * 2),
        ]  # 16

        sequence += [
            nn.Conv2d(ndf * 2, ndf * 4,
                      kernel_size=4, stride=2, padding=1),
            nn.SELU(True),
            DTunnel(2, ndf * 4),
        ]  # 8

        sequence += [
            nn.Conv2d(ndf * 4, ndf * 8,
                      kernel_size=4, stride=2, padding=1),
            nn.SELU(True),
            DTunnel(2, ndf * 8),  # 4
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=0)
        ]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)
