#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 14:54:07 2021

@author: yaoyichen
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as nnf
import sys
import os
import torch
from torch.autograd.functional import jacobian
from torch.autograd import grad
from torch.optim import LBFGS, Adam, SGD
import logging
import time
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
from models.DimChangeModule import DimChange


class DoubleConv(nn.Module):
    def __init__(self, feature_channel, output_channel):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(feature_channel, output_channel, 3, 1, 1, bias=False),
            nn.BatchNorm1d(output_channel),
            nn.ReLU(inplace=True),

            nn.Conv1d(output_channel, output_channel, 3, 1, 1, bias=False),
            nn.BatchNorm1d(output_channel),
            nn.ReLU(inplace=True),

        )

    def forward(self, x):
        return self.conv(x)


class UNET(nn.Module):
    # def __init__(self, feature_channel=20, output_channel=4,
    #              features=[32, 64, 128, 256], signal_length=56, dim_add=0):

    def __init__(self, feature_channel=20, output_channel=4,
                 features=[32, 64, 128, 256]):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # self.signal_length = signal_length
        # self.dim_add = dim_add
        # self.dim_change = None
        # if(self.dim_add > 0):
        #     self.dim_change = DimChange(
        #         signal_length, signal_length + self.dim_add)

        # down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(feature_channel, feature))
            feature_channel = feature

        # up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose1d(feature*2, feature,
                                   kernel_size=2, stride=2,)
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv1d(features[0], output_channel, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            # print(x.shape, skip_connection.shape)
            if x.shape != skip_connection.shape:
                x = nnf.interpolate(x, size=(skip_connection.shape[-1],),
                                    mode='linear', align_corners=False)
                # x = torch.nn.functional.upsample(
                #     x, size=skip_connection.shape[-1], scale_factor=None, mode='linear', align_corners=None)
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)
        x = self.final_conv(x)

        # if(self.dim_change):
        #     x = self.dim_change(x)
        return x


def test():
    # uNet = UNET(feature_channel=20, output_channel=4,
    #             signal_length=56, dim_add=1)
    uNet = UNET(feature_channel=20, output_channel=4)
    x = torch.randn([1000, 20, 57])
    y = uNet(x)
    print(y.shape)

    # ModelUtils.get_parameter_number(uNet)


if __name__ == "__main__":
    test()
