#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 13:31:46 2019

@author: ishwark
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


class ConvAutoEncoder(nn.Module):
    """Simple VAE model: `Convolution-->FullyConnected-->DeConvolution` """
    def __init__(self, in_dim, out_dim):
        super(ConvAutoEncoder, self).__init__()

        k_s = [16, 32, 64]

        self.lin_input_shape = in_dim[0] * in_dim[1] * in_dim[2]

        self.encoder = nn.Sequential(
            nn.Conv2d(in_dim[0], k_s[0], kernel_size=3, stride=1),
            nn.ELU(),
            nn.Dropout(),

            nn.Conv2d(k_s[0], k_s[1], kernel_size=5, stride=2),
            nn.ELU(),
            nn.Dropout(),

            nn.Conv2d(k_s[1], k_s[2], kernel_size=5, stride=1),
            nn.ELU(),
            nn.Dropout(),

            nn.Conv2d(k_s[2], 8, kernel_size=3, stride=1),
            nn.Sigmoid()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1, k_s[0], 3, stride=1),
            nn.ReLU(),

            nn.ConvTranspose2d(k_s[0], k_s[1], 5, stride=1),
            nn.ReLU(),
            nn.Dropout(),

            nn.ConvTranspose2d(k_s[1], k_s[2], 5, stride=2),
            nn.ReLU(),
            nn.Dropout(),

            nn.ConvTranspose2d(k_s[2], out_dim[0], 3, stride=1),
            nn.Sigmoid()
        )

        self.in_lin_sz = self._encoder_out_sz(in_dim)

        self.out_sz = self._decoder_in_sz(out_dim)

        out_sz_lin = self.out_sz[0] * self.out_sz[1]

        self._mu = nn.Linear(self.in_lin_sz, out_sz_lin)

        self._var = nn.Linear(self.in_lin_sz, out_sz_lin)

    def _encoder_out_sz(self, in_dim):
        b_im_dim = [1] + in_dim
        enc = self.encoder(torch.Tensor(torch.ones(b_im_dim)))
        return enc.view(1, -1).shape[1]

    def _decoder_in_sz(self, out_dim):
        out = [out_dim[-2], out_dim[-1]]
        for d in reversed(self.decoder):
            if isinstance(d, nn.ConvTranspose2d):
                out[0] = (out[0] - math.ceil(d.kernel_size[0]/2)) / d.stride[0]
                out[1] = (out[1] - math.ceil(d.kernel_size[1]/2)) / d.stride[1]

        return [int(i) for i in out]

    def forward(self, x):
        enc = self.encoder(x)
        enc_lin = enc.view(enc.size(0), -1)
        mu = self._mu(enc_lin)

        if not self.training:
            return mu

        var = self._var(enc_lin)

        out = self._reparameterize(mu, var)

        lin2d = out.reshape([-1, 1, self.out_sz[0], self.out_sz[1]])

        out = self.decoder(lin2d)[:, :, 1:, 1:]
        return out, mu, var

    def _reparameterize(self, mu, logvar):
        """ from https://sergioskar.github.io/Autoencoder/"""
        std = (logvar * 0.5).exp_()
        eps = std.data.new(std.size()).normal_()
        return eps.mul(std).add_(mu)

    def loss_function(self, recon_x, x, mu=None, logvar=None):
        """ from https://sergioskar.github.io/Autoencoder/"""
        bce = F.binary_cross_entropy(recon_x, x)
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kld /= x.shape[0] * self.in_lin_sz
        return bce, kld
