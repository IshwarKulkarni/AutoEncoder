#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 13:31:46 2019

@author: ishwark
"""
import torch
from torch import nn
from torch.nn import functional as F
import math


class ConvAutoEncoder(nn.Module):
    def __init__(self, in_dim=[3, 32, 32], out_dim=[3, 32, 32]):
        super(ConvAutoEncoder, self).__init__()
        
        k_s = [3, 6, 9]
        k_s = [12, 24, 48]
               
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
        
        self.mu = nn.Linear(self.in_lin_sz, out_sz_lin)

        self.var = nn.Linear(self.in_lin_sz, out_sz_lin)
        
    def _encoder_out_sz(self, in_dim):
        b_im_dim = [1] + in_dim
        enc = self.encoder(torch.Tensor(torch.ones(b_im_dim)))
        return enc.view(1, -1).shape[1]
    
    def _decoder_in_sz(self, out_dim):
        out = [out_dim[-2], out_dim[-1]]
        for d in reversed(self.decoder):
            if type(d) == nn.ConvTranspose2d:
                out[0] = (out[0] - math.ceil(d.kernel_size[0]/2))/ d.stride[0]
                out[1] = (out[1] - math.ceil(d.kernel_size[1]/2))/ d.stride[1]
 
        return [int(i) for i in out]
        
    
    def forward(self, x):        
        enc   = self.encoder(x)
        encL  = enc.reshape(enc.size(0), -1)        
        muV   = self.mu(encL)
        varV  = self.var(encL)
        
        out   = self._reparameterize(muV, varV)
        
        lin2d = out.reshape([-1, 1, self.out_sz[0], self.out_sz[1]])
        
        out = self.decoder(lin2d)[:, :, 1:, 1:]
        return out, muV, varV           
        
        print("enc", enc.shape)
        print("out", out.shape)
        print("out", out.shape) 
        print("mu", muV.shape)
        
    """ https://sergioskar.github.io/Autoencoder/
        These 2 methods from above location
    """    
    def _reparameterize(self, mu, logvar):
        if self.training: 
            std = (logvar * 0.5).exp_()
            eps = std.data.new(std.size()).normal_()
            return eps.mul(std).add_(mu)        
        else:
            return mu
        
    def loss_function(self, recon_x, x, mu=None, logvar=None):
        BCE = F.binary_cross_entropy(recon_x, x)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        KLD /= x.shape[0] * 1024  * x.shape[1] 
        return BCE,  KLD
