#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 11:07:16 2021

@author: yaoyichen
"""
import torch
import torch.nn as nn


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, in_modes):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.in_modes = in_modes

        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.in_modes, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,
                             x.size(-1)//2 + 1,  device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.in_modes] = self.compl_mul1d(
            x_ft[:, :, : self.in_modes], self.weights1)

        # Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x


class SpectralConv1dEasy(nn.Module):
    def __init__(self, in_channels, out_channels, in_modes):
        super(SpectralConv1dEasy, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.in_modes = in_modes

        self.scale = (1 / (in_channels*out_channels))
        self.weights1_real = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.in_modes))
        self.weights1_image = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.in_modes))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,
                             x.size(-1)//2 + 1,  device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.in_modes] = self.compl_mul1d(
            x_ft[:, :, : self.in_modes],
            torch.complex(self.weights1_real, self.weights1_image))

        # Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x


class SpectralConv2d(nn.Module):
    """
    do not use this version of SpectralConv.
    it is not good for parallel
    """

    def __init__(self, in_channels, out_channels, in_modes, out_modes):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_modes = in_modes
        self.out_modes = out_modes

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.in_modes, self.out_modes, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.in_modes, self.out_modes, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        # print(input.shape, weights.shape)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)
        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2),
                             x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.in_modes, :self.out_modes] = \
            self.compl_mul2d(
                x_ft[:, :, :self.in_modes, :self.out_modes], self.weights1)
        out_ft[:, :, -self.in_modes:, :self.out_modes] = \
            self.compl_mul2d(
                x_ft[:, :, -self.in_modes:, :self.out_modes], self.weights2)

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class SpectralConv2dEasy(nn.Module):
    def __init__(self, in_channels, out_channels, in_modes, out_modes):
        super(SpectralConv2dEasy, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_modes = in_modes
        self.out_modes = out_modes

        self.scale = (1 / (in_channels * out_channels))
        self.weights1_real = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.in_modes, self.out_modes))
        self.weights1_image = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.in_modes, self.out_modes))

        self.weights2_real = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.in_modes, self.out_modes))

        self.weights2_image = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.in_modes, self.out_modes))
    # Complex multiplication

    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        # print(input.shape, weights.shape)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)
        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2),
                             x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.in_modes, :self.out_modes] = \
            self.compl_mul2d(
                x_ft[:, :, :self.in_modes, :self.out_modes], torch.complex(self.weights1_real, self.weights1_image))
        out_ft[:, :, -self.in_modes:, :self.out_modes] = \
            self.compl_mul2d(
                x_ft[:, :, -self.in_modes:, :self.out_modes], torch.complex(self.weights2_real, self.weights2_image))

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class SVDOperation2d(nn.Module):
    def __init__(self, in_channels, out_channels, in_modes, out_modes):
        """
        number of svd modes to perform
        """
        super(SVDOperation2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_modes = in_modes
        self.out_modes = out_modes
        self.scale = (1 / (in_channels * out_channels))

        self.weights_u = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.in_modes, self.out_modes))

        self.weights_s = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.in_modes, self.out_modes))

        self.weights_v = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.in_modes, self.out_modes))

    def forward(self, x):
        u, s, v = torch.svd(x)
        uu = u[..., 0:self.in_modes]
        ss = s[..., 0: self.in_modes]
        vv = v[..., 0:self.in_modes]

        uu = torch.einsum("bcij,cdjk->bdik", uu, self.weights_u)
        ss = torch.einsum("bcj,cdjk->bdk", ss, self.weights_s)
        vv = torch.einsum("bcij,cdjk->bdik", vv, self.weights_v)

        x_reconstruct = torch.einsum("bcij,bcjk,bctk->bcit", uu,
                                     torch.diag_embed(ss), vv)

        return x_reconstruct


class SVDOperation2d_2stage(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, in_modes, out_modes):
        """
        number of svd modes to perform
        """
        super(SVDOperation2d_2stage, self).__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels
        self.in_modes = in_modes
        self.out_modes = out_modes
        self.scale = (1 / (in_channels * out_channels))

        self.weights_u = nn.Parameter(
            self.scale * torch.rand(in_channels, in_channels, self.in_modes, self.out_modes))

        self.weights_s_u = nn.Parameter(
            self.scale * torch.rand(in_channels, in_channels, self.in_modes, self.out_modes))

        self.weights_s_v = nn.Parameter(
            self.scale * torch.rand(in_channels, in_channels, self.in_modes, self.out_modes))

        self.weights_v = nn.Parameter(
            self.scale * torch.rand(in_channels, in_channels, self.in_modes, self.out_modes))

    def forward(self, x):
        u, s, v = torch.svd(x)
        uu = u[..., 0:self.in_modes]
        ss = s[..., 0: self.in_modes]
        vv = v[..., 0:self.out_modes]

        uu = torch.einsum("bcij,cdjk->bdik", uu, self.weights_u)
        ss = torch.einsum("bcj,cdjk->bdk", ss, self.weights_s_u)

        x = torch.einsum("bcij,bcjk,bctk->bcit", uu,
                         torch.diag_embed(ss), vv)

        u, s, v = torch.svd(x)

        uu = u[..., 0:self.out_modes]
        ss = s[..., 0: self.in_modes]
        vv = v[..., 0:self.in_modes]

        ss = torch.einsum("bcj,cdjk->bdk", ss, self.weights_s_v)
        vv = torch.einsum("bcij,cdjk->bdik", vv, self.weights_v)

        x_reconstruct = torch.einsum("bcij,bcjk,bctk->bcit", uu,
                                     torch.diag_embed(ss), vv)

        return x_reconstruct



def testFNO1d():
    net = SpectralConv1d(
        in_channels=15, out_channels=12, in_modes=8)
    y = net(torch.randn(1000, 15, 32))
    print(y.size())


def testFNO2d():
    net = SpectralConv2d(
        in_channels=15, out_channels=12, in_modes=4, out_modes=4)
    y = net(torch.randn(1000, 15, 32, 32))

    print(y.size())


def testSVD2d():
    a = torch.randn(15, 10, 32, 29)
    net = SVDOperation2d(in_channels=10, out_channels=12,
                         in_modes=5, out_modes=6)

    net = SVDOperation2d_2stage(in_channels=10, in_modes=5, out_modes=6)

    y = net(a)
    print(y.shape)


def testWRT1d():
    net = WaveNet1dConti(in_channels=12, out_channels=12)

    y = net(torch.randn(1000, 12, 57))
    print(y.size())


if __name__ == "__main__":
    testWRT1d()
