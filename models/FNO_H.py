
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 14:54:07 2021

@author: yaoyichen
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import sys
sys.path.append("..")
from models.DimChangeModule import DimChange
from models.Neural_Operators import SpectralConv2d, SpectralConv2dEasy, SpectralConv1d, SpectralConv1dEasy


class FNO1d(nn.Module):
    def __init__(self, feature_channel,
                 output_channel, modes, width):

        # def __init__(self, feature_channel,
        #              output_channel, modes, width, signal_length=56, dim_add=0):
        super(FNO1d, self).__init__()
#
        # self.signal_length = signal_length
        self.padding = 2  # pad the domain if input is non-periodic
        # self.dim_add = dim_add
        # self.dim_change = None
        # if(self.dim_add > 0):
        #     self.dim_change = DimChange(
        #         signal_length + 2*self.padding, signal_length + self.dim_add)

        self.feature_channel = feature_channel
        self.output_channel = output_channel

        self.modes1 = modes
        self.width = width

        # input channel is 2: (a(x), x)
        self.fc0 = nn.Linear(self.feature_channel, self.width)

        self.conv0 = SpectralConv1dEasy(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1dEasy(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1dEasy(self.width, self.width, self.modes1)
        self.conv3 = SpectralConv1dEasy(self.width, self.width, self.modes1)

        # self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        # self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        # self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        # self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)

        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, self.width)
        self.fc2 = nn.Linear(self.width, self.output_channel)

    def forward(self, x):
        x = torch.permute(x, (0, 2, 1))
        x = self.fc0(x)
        x = x.permute(0, 2, 1)
        # pad the domain if input is non-periodic

        x = F.pad(x, [self.padding, self.padding])

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)

        x = torch.permute(x, (0, 2, 1))

        # if(self.dim_change):
        #     x = self.dim_change(x)
        x = x[:, :, self.padding: -self.padding]
        return x


def test():
    net = FNO1d(feature_channel=20,
                output_channel=4, modes=20, width=128)
    print(net)
    y = net(torch.randn(1000, 20, 57))
    print(y.size())


if __name__ == "__main__":
    test()
