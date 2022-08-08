import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np
import h5py
import scipy.io
import matplotlib.pyplot as plt
from timeit import default_timer
import sys
import math

import operator
from functools import reduce

from timeit import default_timer
# from utilities3 import *

torch.manual_seed(0)
np.random.seed(0)

################################################################
# lowrank layer
################################################################


# A simple feedforward neural network
class DenseNet(torch.nn.Module):
    def __init__(self, layers, nonlinearity, out_nonlinearity=None, normalize=False):
        super(DenseNet, self).__init__()

        self.n_layers = len(layers) - 1

        assert self.n_layers >= 1

        self.layers = nn.ModuleList()

        for j in range(self.n_layers):
            self.layers.append(nn.Linear(layers[j], layers[j+1]))

            if j != self.n_layers - 1:
                if normalize:
                    self.layers.append(nn.BatchNorm1d(layers[j+1]))

                self.layers.append(nonlinearity())

        if out_nonlinearity is not None:
            self.layers.append(out_nonlinearity())

    def forward(self, x):
        for _, l in enumerate(self.layers):
            x = l(x)

        return x


class LowRank1d(nn.Module):
    def __init__(self, in_channels, out_channels, s, width, rank=1):
        super(LowRank1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.rank = rank

        self.phi = DenseNet(
            [56, 64, 256, width*width*rank], torch.nn.ReLU)
        self.psi = DenseNet(
            [56, 64, 256, width*rank*rank], torch.nn.ReLU)

    def forward(self, v, a):
        print(v.shape, a.shape)
        # a (batch, n, 2)
        # v (batch, n, f)
        batch_size = v.shape[0]
        temp = self.phi(a)
        print(temp.shape)
        phi_eval = temp.reshape(batch_size,
                                self.in_channels, self.out_channels, self.rank)

        print(phi_eval.shape)
        psi_eval = self.psi(a).reshape(batch_size, self.out_channels,
                                       self.in_rank, self.out_rank)

        print(v.shape, phi_eval.shape,  psi_eval.shape)
        # print(psi_eval.shape, v.shape, phi_eval.shape)
        # v = torch.einsum('bnoir,bni,bmoir->bmo',
        v = torch.einsum('bnmr,bnr,bmrq->bmq',
                         phi_eval, v, psi_eval)

        return v


class MyNet(torch.nn.Module):
    def __init__(self, s, width=32, rank=4):
        super(MyNet, self).__init__()
        self.s = s
        self.width = width
        self.rank = rank

        self.fc0 = nn.Linear(56, self.width)

        self.net1 = LowRank1d(self.width, self.width, s, width, rank=self.rank)
        self.net2 = LowRank1d(self.width, self.width, s, width, rank=self.rank)
        self.net3 = LowRank1d(self.width, self.width, s, width, rank=self.rank)
        self.net4 = LowRank1d(self.width, self.width, s, width, rank=self.rank)
        self.w1 = nn.Linear(self.width, self.width)
        self.w2 = nn.Linear(self.width, self.width)
        self.w3 = nn.Linear(self.width, self.width)
        self.w4 = nn.Linear(self.width, self.width)

        self.bn1 = torch.nn.BatchNorm1d(self.width)
        self.bn2 = torch.nn.BatchNorm1d(self.width)
        self.bn3 = torch.nn.BatchNorm1d(self.width)
        self.bn4 = torch.nn.BatchNorm1d(self.width)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, v):
        print("1"*20)
        print(v.shape)
        batch_size, n = v.shape[0], v.shape[1]
        a = v.clone()
        print(a.shape)

        v = self.fc0(v)
        print("2"*20)
        print(v.shape)

        v2 = self.w1(v)
        print("2.2"*20)
        print(v2.shape)

        v1 = self.net1(v, a)
        print("2.1"*20)
        print(v1.shape)

        v = v1+v2
        v = self.bn1(v.reshape(-1, self.width)).view(batch_size, n, self.width)
        v = F.relu(v)

        print("3"*20)
        print(v.shape)

        v1 = self.net2(v, a)
        v2 = self.w2(v)
        v = v1+v2
        v = self.bn2(v.reshape(-1, self.width)).view(batch_size, n, self.width)
        v = F.relu(v)

        v1 = self.net3(v, a)
        v2 = self.w3(v)
        v = v1+v2
        v = self.bn3(v.reshape(-1, self.width)).view(batch_size, n, self.width)
        v = F.relu(v)

        v1 = self.net4(v, a)
        v2 = self.w4(v)
        v = v1+v2
        v = self.bn4(v.reshape(-1, self.width)).view(batch_size, n, self.width)

        v = self.fc1(v)
        v = F.relu(v)
        v = self.fc2(v)

        return v.squeeze()

    def count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))

        return c


def test():
    net = MyNet(s=10.0, width=8, rank=10)
    print(net)
    y = net(torch.randn(1000, 20, 56))
    print(y.size())


if __name__ == "__main__":
    test()
