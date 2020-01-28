#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 13:31:46 2019

@author: ishwark
"""
import math

import copy

import torch
from torch import nn
from torch.nn import functional as F

# pylint: disable=E1101
# pylint: disable=R0902

class ResnetBlocks(nn.Module):
    """ Residual blocks"""
    def __init__(self, conv: bool, in_ch: int, out_ch: int, activation: nn.Module,
                 num_blocks=2, k_sz=3, stride=1, drop=True):
        super(ResnetBlocks, self).__init__()
        paddding = k_sz // 2
        input_layer = [nn.Conv2d(in_ch, out_ch, k_sz, stride, padding=paddding) if conv else \
                       nn.ConvTranspose2d(in_ch, out_ch, k_sz, stride, padding=paddding),
                       activation,
                       nn.BatchNorm2d(out_ch)]

        b_blocks = [nn.Conv2d(out_ch, out_ch, k_sz, padding=paddding) if conv else\
                    nn.ConvTranspose2d(out_ch, out_ch, k_sz, padding=paddding),
                    activation]

        if drop:
            input_layer.append(nn.Dropout())

        blocks = []
        for _ in range(0, num_blocks):
            blocks += [*copy.deepcopy(b_blocks)]

        if drop:
            b_blocks.append(nn.Dropout())

        all_blocks = [*input_layer, *blocks, nn.BatchNorm2d(out_ch)]

        self.block = nn.Sequential(*all_blocks)

        self.shortcut = nn.Conv2d(in_ch, out_ch, k_sz, stride, padding=paddding) if conv else \
                                  nn.ConvTranspose2d(in_ch, out_ch, k_sz, stride, padding=paddding)

    def forward(self, x):
        return self.block(x)  # + self.shortcut(x) if self.shortcut else x


def _reparameterize(mean, logvar, label=None):
    """ from https://sergioskar.github.io/Autoencoder/"""
    std = (logvar * 0.5).exp_()
    eps = std.data.new(std.size()).normal_()
    z = mean + eps * std
    z_c = torch.cat ((z, label), dim=1)
    return z if label is None else z_c


class ConvAutoEncoder(nn.Module):
    """Simple VAE model: `Convolution-->FullyConnected-->DeConvolution` """

    def __init__(self, in_dim, out_dim, z_size=6, c_size=0):
        super(ConvAutoEncoder, self).__init__()

        base = [12, 18, 18, 36]
        ek_s = [int(i)*2 for i in base]
        dk_s = ek_s
        dk_s.reverse()

        in_ch, out_ch = in_dim[0], out_dim[0]

        enc_out_z = 3

        dec_out_z = 1

        self.lin_input_shape = in_dim[0] * in_dim[1] * in_dim[2]

        self.encoder = nn.Sequential(
            ResnetBlocks(True, in_ch, ek_s[0], nn.ELU(), stride=3),
            ResnetBlocks(True, ek_s[0], ek_s[1], nn.ReLU(), stride=2),
            ResnetBlocks(True, ek_s[1], ek_s[2], nn.ReLU(), stride=2),
            ResnetBlocks(True, ek_s[2], ek_s[3], nn.ReLU()),
            ResnetBlocks(True, ek_s[3], enc_out_z, nn.ReLU()),
            nn.Sigmoid()
        )

        self.decoder = nn.Sequential(
            ResnetBlocks(False, dec_out_z, dk_s[0], nn.ELU()),
            ResnetBlocks(False, dk_s[0], dk_s[1], nn.ReLU()),
            ResnetBlocks(False, dk_s[1], dk_s[2], nn.ReLU(), stride=2),
            ResnetBlocks(False, dk_s[2], dk_s[3], nn.ReLU(), stride=2),
            ResnetBlocks(False, dk_s[3], out_ch, nn.ReLU()),
            nn.Sigmoid()
        )

        self.in_lin_sz = self._encoder_out_sz(in_dim)

        self.out_sz = self._decoder_in_sz(out_dim)

        out_sz_lin = self.out_sz[0] * self.out_sz[1]

        self._mu = nn.Linear(self.in_lin_sz, z_size)

        self._var = nn.Linear(self.in_lin_sz, z_size)

        self._decode_embedded = nn.Linear(z_size + c_size, out_sz_lin)

        self.fake_training = False


    def _encoder_out_sz(self, in_dim):
        b_im_dim = [1] + in_dim
        temp = self.encoder(torch.ones(b_im_dim))
        return temp.view(1, -1).shape[1]

    def _decoder_in_sz(self, out_dim):
        out = [out_dim[-2], out_dim[-1]]
        for d in reversed(self.decoder):
            if isinstance(d, nn.ConvTranspose2d):
                out[0] = (out[0] - math.ceil(d.kernel_size[0] / 2)) // d.stride[0]
                out[1] = (out[1] - math.ceil(d.kernel_size[1] / 2)) // d.stride[1]
            elif isinstance(d, ResnetBlocks):
                for dd in d.block:
                    if isinstance(dd, nn.ConvTranspose2d):
                        out[0] = 1 + math.ceil((out[0] + 2 * dd.padding[0] - dd.kernel_size[0]) / dd.stride[0])
                        out[1] = 1 + math.ceil((out[1] + 2 * dd.padding[1] - dd.kernel_size[1]) / dd.stride[1])

        return [int(i) for i in out]

    def forward(self, x, c=None):
        enc = self.encoder(x)
        enc_lin = enc.view(enc.size(0), -1)
        mean = F.normalize(self._mu(enc_lin))

        if not self.training and not self.fake_training:
            return mean

        var = self._var(enc_lin)

        out = _reparameterize(mean, var, c)

        out = self._decode_embedded(out)

        lin2d = out.reshape([-1, 1, self.out_sz[0], self.out_sz[1]])

        out = self.decoder(lin2d)
        return out, mean, var

    def conv_encode(self, image, label):
        enc = self.encoder(image)
        enc_lin = enc.view(enc.size(0), -1)
        return enc, F.normalize( self._mu(enc_lin))

    def loss_function(self, recon_x, image, mean=None, logvar=None):
        """Kulback Lieber based loss """
        bce = F.binary_cross_entropy(recon_x[:, :, 1:, 1:], image)
        kld = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        kld /= image.shape[0] * self.lin_input_shape
        return bce, kld
