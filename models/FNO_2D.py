
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
from models.Neural_Operators import SpectralConv2d, SpectralConv1d, SpectralConv2dEasy
################################################################
# fourier layer
################################################################


class FNO2d(nn.Module):
    # def __init__(self, modes1, modes2,  width, in_channels, out_channels, signal_length, dim_add=0):

    def __init__(self, modes1, modes2,  width, in_channels, out_channels):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 3  # pad the domain if input is non-periodic
        # input channel is 3: (a(x, y), x, y)
        self.fc0 = nn.Linear(1, self.width)

        self.conv0 = SpectralConv2dEasy(
            self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2dEasy(
            self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2dEasy(
            self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2dEasy(
            self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, self.width)
        self.fc2 = nn.Linear(self.width, 1)

        self.fc3 = nn.Linear(in_channels, out_channels)

        # self.fc4 = nn.Linear(signal_length, signal_length + dim_add)

    # def change_dim(self, x):

    def forward(self, x):
        x = x.unsqueeze(dim=1)

        x = x.permute(0, 2, 3, 1)
        x = self.fc0(x)

        x = x.permute(0, 3, 1, 2)
        x = F.pad(x, [0, self.padding, 0, self.padding])

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

        x = x[..., :-self.padding, :-self.padding]
        x = x.permute(0, 2, 3, 1)

        x = self.fc1(x)
        x = F.gelu(x)

        x = self.fc2(x)
        x = x.squeeze()
        x = F.relu(x)

        x = x.permute(0, 2, 1)
        x = self.fc3(x)
        x = x.permute(0, 2, 1)

        return x


def test():
    # net = FNO2d(modes1=1, modes2=2,  width=16, in_channels=34,
    #             out_channels=4, signal_length=57, dim_add=0)
    net = FNO2d(modes1=1, modes2=2,  width=16, in_channels=34,
                out_channels=4)
    print(net)
    import time
    time1 = time.time()
    y = net(torch.randn(20, 34, 57))
    print(y.size())
    print(time.time() - time1)


def testBlock():
    net = SpectralConv2d(
        in_channels=15, out_channels=12, in_modes=4, out_modes=4)
    y = net(torch.randn(1000, 15, 32, 32))
    print(y.size())


if __name__ == "__main__":
    test()
