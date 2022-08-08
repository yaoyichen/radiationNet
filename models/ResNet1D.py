
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 14:54:07 2021

@author: yaoyichen
"""


import torch
import torch.nn as nn
import sys
sys.path.append("..")
from models.DimChangeModule import DimChange


class block(nn.Module):
    def __init__(
        self, in_channels, intermediate_channels, identity_downsample=None, stride=1
    ):
        super(block, self).__init__()
        self.expansion = 1
        self.conv1 = nn.Conv1d(
            in_channels, intermediate_channels, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn1 = nn.BatchNorm1d(intermediate_channels)
        self.conv2 = nn.Conv1d(
            intermediate_channels,
            intermediate_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm1d(intermediate_channels)
        self.conv3 = nn.Conv1d(
            intermediate_channels,
            intermediate_channels * self.expansion,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        self.bn3 = nn.BatchNorm1d(intermediate_channels * self.expansion)
        self.relu = nn.ReLU()
        # self.identity_downsample = identity_downsample
        self.stride = stride

    def forward(self, x):
        identity = x.clone()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        x += identity
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    # def __init__(self, layers, feature_channel, output_channel, intermediate_channel=64, signal_length=56, dim_add=0):
    def __init__(self, layers, feature_channel, output_channel, intermediate_channel=64):

        super(ResNet, self).__init__()
        self.in_channels = intermediate_channel
        self.conv1 = nn.Conv1d(
            feature_channel, intermediate_channel, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(intermediate_channel)
        self.relu = nn.ReLU()

        # self.signal_length = signal_length
        # self.dim_add = dim_add
        # self.dim_change = None
        # if(self.dim_add > 0):
        #     self.dim_change = DimChange(
        #         signal_length, signal_length + self.dim_add)

        # Essentially the entire ResNet architecture are in these 4 lines below
        self.layer1 = self._make_layer(
            block, layers[0], intermediate_channels=intermediate_channel, stride=1
        )

        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv_final = nn.Conv1d(
            intermediate_channel, output_channel, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.conv_final(x)
        # if(self.dim_change):
        #     x = self.dim_change(x)
        return x

    def _make_layer(self, block, num_residual_blocks, intermediate_channels, stride):
        identity_downsample = None
        layers = []

        # The expansion size is always 4 for ResNet 50,101,152
        self.in_channels = intermediate_channels

        # For example for first resnet layer: 256 will be mapped to 64 as intermediate layer,
        # then finally back to 256. Hence no identity downsample is needed, since stride = 1,
        # and also same amount of channels.
        for i in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, intermediate_channels))

        return nn.Sequential(*layers)


def test():
    layers = [10]
    # net = ResNet(layers, feature_channel=20,
    #              output_channel=4, intermediate_channel=64,  signal_length=56, dim_add=0)
    net = ResNet(layers, feature_channel=20,
                 output_channel=4, intermediate_channel=64)
    print(net)
    y = net(torch.randn(1000, 20, 57))
    print(y.size())


if __name__ == "__main__":
    test()
