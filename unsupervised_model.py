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
    """Simple VAE model: `Convolution-->FullyConnected-->DeConvolution`
    """
    def __init__(self, in_dim, out_dim, embedding_size=5):
        super(ConvAutoEncoder, self).__init__()

        ek_s = [64, 96, 128, 192]
        dk_s = [32, 64, 96, 128]

        self.lin_input_shape = in_dim[0] * in_dim[1] * in_dim[2]

        self.encoder = nn.Sequential(
            nn.Conv2d(in_dim[0], ek_s[0], kernel_size=3, stride=1),
            nn.ELU(),

            nn.Conv2d(ek_s[0], ek_s[1], kernel_size=3, stride=2),
            nn.ReLU(),

            nn.Conv2d(ek_s[1], ek_s[2], kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Dropout(),

            nn.Conv2d(ek_s[2], ek_s[3], kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Dropout(),

            nn.Conv2d(ek_s[3], 16, kernel_size=3, stride=1),
            nn.Sigmoid()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1, dk_s[0], kernel_size=3, stride=1),
            nn.ELU(),

            nn.ConvTranspose2d(dk_s[0], dk_s[1], kernel_size=3, stride=1),
            nn.ReLU(),

            nn.ConvTranspose2d(dk_s[1], dk_s[2], kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Dropout(),

            nn.ConvTranspose2d(dk_s[2], dk_s[3], kernel_size=4, stride=1),
            nn.ReLU(),
            nn.Dropout(),

            nn.ConvTranspose2d(dk_s[3], out_dim[0], kernel_size=3, stride=1),
            nn.Sigmoid()
        )

        self.in_lin_sz = self._encoder_out_sz(in_dim)

        self.out_sz = self._decoder_in_sz(out_dim)

        out_sz_lin = self.out_sz[0] * self.out_sz[1]

        self._mu = nn.Linear(self.in_lin_sz, embedding_size)

        self._var = nn.Linear(self.in_lin_sz, embedding_size)

        self._decode_embedded = nn.Linear(embedding_size, out_sz_lin)


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

        out = self._decode_embedded(out)

        lin2d = out.reshape([-1, 1, self.out_sz[0], self.out_sz[1]])

        out = self.decoder(lin2d)
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
        kld /= x.shape[0] * self.lin_input_shape
        return bce, kld
