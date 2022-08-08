
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
from models.Neural_Operators import SVDOperation2d


# class SVDConv2d(nn.Module):
#     def __init__(self, in_channels, out_channels, modes1, modes2):
#         super(SVDConv2d, self).__init__()

#     def forward(self, input):
#         """
#         B C H W
#         """


class SVD2D(nn.Module):
    def __init__(self,  in_modes, out_modes,  width, in_channels, out_channels, signal_length, dim_add):
        super(SVD2D, self).__init__()

        """
        """
        self.in_modes = in_modes
        self.out_modes = out_modes
        self.width = width
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.fc0 = nn.Linear(1, self.width)

        self.block0 = SVDOperation2d(
            self.width, self.width, self.in_modes, self.out_modes)
        self.block1 = SVDOperation2d(
            self.width, self.width, self.in_modes, self.out_modes)
        self.block2 = SVDOperation2d(
            self.width, self.width, self.in_modes, self.out_modes)
        self.block3 = SVDOperation2d(
            self.width, self.width, self.in_modes, self.out_modes)

        self.fc1 = nn.Linear(self.width, self.width)
        self.fc2 = nn.Linear(self.width, 1)
        self.fc3 = nn.Linear(in_channels, out_channels)
        self.fc4 = nn.Linear(signal_length, signal_length + dim_add)

    def forward(self, x):
        # x = x + 1.0e-2*torch.randn(x.shape)
        print(x.shape)
        x = x.unsqueeze(dim=1)
        x = x.permute(0, 2, 3, 1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        x = self.block0(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)

        x = x.squeeze()

        x = x.permute(0, 2, 1)
        x = self.fc3(x)
        x = x.permute(0, 2, 1)

        x = self.fc4(x)

        return x


def test():
    import time
    time1 = time.time()
    net = SVD2D(in_modes=12, out_modes=12, width=24,
                in_channels=29, out_channels=4, signal_length=56, dim_add=1)

    print(net)
    time1 = time.time()
    y = net(torch.randn(1000, 29, 56))
    print(time.time() - time1)
    print(y.size())


if __name__ == "__main__":
    test()
