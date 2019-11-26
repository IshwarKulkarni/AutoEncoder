#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 13:31:46 2019

@author: ishwark
"""
import math

import torch
import copy
from torch import nn
from torch.nn import functional as F

class ResnetBlocks(nn.Module):
    """"""
    def __init__(self, conv:bool, in_ch:int, out_ch:int, activation:nn.Module,
                 num_blocks=2, k_sz=3, stride=1, drop=True):
        super(ResnetBlocks, self).__init__()
        paddding = k_sz//2
        input_layer = [
            nn.Conv2d(in_ch, out_ch, k_sz, stride, padding=paddding) if conv else\
            nn.ConvTranspose2d(in_ch, out_ch, k_sz, stride, padding=paddding),
            activation,
            nn.BatchNorm2d(out_ch)
        ]

        b_blocks = [
            nn.Conv2d(out_ch, out_ch, k_sz, padding=paddding) if conv else\
            nn.ConvTranspose2d(out_ch, out_ch, k_sz, padding=paddding),
            activation
        ]
        
        if drop:
            input_layer.append(nn.Dropout())        

        blocks = []
        for _ in range(0, num_blocks):
            blocks+= [  *copy.deepcopy(b_blocks) ]

        if drop:
            b_blocks.append(nn.Dropout())

        all_blocks = [*input_layer, *blocks, nn.BatchNorm2d(out_ch)]

        self.block = nn.Sequential(*all_blocks)

        self.shortcut = nn.Conv2d(in_ch, out_ch, k_sz, stride, padding=paddding) if conv else\
                        nn.ConvTranspose2d(in_ch, out_ch, k_sz, stride, padding=paddding)

    def forward(self, x):
        b = self.block(x)
        s = self.shortcut(x) if self.shortcut else x
        return b + s

class ConvAutoEncoder(nn.Module):
    """Simple VAE model: `Convolution-->FullyConnected-->DeConvolution`
    """
    def __init__(self, in_dim, out_dim, embedding_size=6):
        super(ConvAutoEncoder, self).__init__()

        ek_s = [64, 64, 64, 80, 128]

        in_ch , out_ch = in_dim[0], out_dim[0]

        self.lin_input_shape = in_dim[0] * in_dim[1] * in_dim[2]       

        self.encoder = nn.Sequential(
            ResnetBlocks(True, in_ch,   ek_s[0], nn.ELU(), stride=2),
            ResnetBlocks(True, ek_s[0], ek_s[1], nn.ReLU(), stride=2),
            ResnetBlocks(True, ek_s[1], ek_s[2], nn.ReLU(), stride=2),
            ResnetBlocks(True, ek_s[2], ek_s[3], nn.ReLU(), stride=2),
            ResnetBlocks(True, ek_s[3], ek_s[4], nn.ReLU()),
            ResnetBlocks(True, ek_s[4], 16,   nn.Sigmoid()),

            nn.Sigmoid()
        )

        self.decoder = nn.Sequential(
            ResnetBlocks(False, 1,       ek_s[0], nn.ELU()),
            ResnetBlocks(False, ek_s[0], ek_s[1], nn.ReLU()),
            ResnetBlocks(False, ek_s[1], ek_s[2], nn.ReLU()),
            ResnetBlocks(False, ek_s[2], ek_s[3], nn.ReLU(), stride=2),
            ResnetBlocks(False, ek_s[3], ek_s[4], nn.ReLU(), stride=2),
            ResnetBlocks(False, ek_s[4], out_ch,  nn.Sigmoid()),

            nn.Sigmoid()
        )

        self.in_lin_sz = self._encoder_out_sz(in_dim)

        self.out_sz = self._decoder_in_sz(out_dim)

        out_sz_lin = self.out_sz[0] * self.out_sz[1]

        self._mu = nn.Linear(self.in_lin_sz, embedding_size)

        self._var = nn.Linear(self.in_lin_sz, embedding_size)

        self._decode_embedded = nn.Linear(embedding_size, out_sz_lin)

        self.fake_training = False


    def _encoder_out_sz(self, in_dim):
        b_im_dim = [1] + in_dim
        t = self.encoder( torch.ones(b_im_dim) )
        return t.view(1, -1).shape[1]

    def _decoder_in_sz(self, out_dim):
        out = [out_dim[-2], out_dim[-1]]
        for d in reversed(self.decoder):
            if isinstance(d, nn.ConvTranspose2d):
                out[0] = (out[0] - math.ceil(d.kernel_size[0]/2)) // d.stride[0]
                out[1] = (out[1] - math.ceil(d.kernel_size[1]/2)) // d.stride[1]
            elif isinstance(d, ResnetBlocks):
                for dd in d.block:
                    if isinstance(dd, nn.ConvTranspose2d):
                        out[0] = 1 + math.ceil ( (out[0] + 2 * dd.padding[0] - dd.kernel_size[0]) / dd.stride[0])
                        out[1] = 1 + math.ceil ( (out[1] + 2 * dd.padding[1] - dd.kernel_size[1]) / dd.stride[1])
                    

        return [int(i) for i in out]

    def forward(self, x):
        enc = self.encoder(x)
        enc_lin = enc.view(enc.size(0), -1)
        mu = F.normalize( self._mu(enc_lin) )

        if not self.training and not self.fake_training:
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
        bce = F.binary_cross_entropy(recon_x[:, :, 1:, 1:], x)
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kld /= x.shape[0] * self.lin_input_shape
        return bce, kld
