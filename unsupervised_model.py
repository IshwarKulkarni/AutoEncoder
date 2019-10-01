#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 13:31:46 2019

@author: ishwark
"""
import torch
from torch import nn
from torch.nn import functional as F

class ConvAutoEncoder(nn.Module):
    def __init__(self):
        super(ConvAutoEncoder, self).__init__()
        
        k_s = [3, 6, 9]
       # k_s = [ 8, 16, 32 ]
               
        self.convEnc = nn.Sequential(
            nn.Conv2d(3, k_s[0], kernel_size=3, stride=1),
            nn.ELU(),
            nn.Dropout(),
            
            nn.Conv2d(k_s[0], k_s[1], kernel_size=5, stride=2),
            nn.ELU(),
            nn.Dropout(),
            
            nn.Conv2d(k_s[1], k_s[2], kernel_size=5, stride=1),
            nn.ELU(),
             
            nn.Conv2d(k_s[2], 8, kernel_size=3, stride=1),
            nn.Sigmoid()
            )
    
        self.mu = nn.Linear(392, 64)
        
        self.var = nn.Linear(392, 64)
        
        self.convDec = nn.Sequential(
            nn.ConvTranspose2d(1, k_s[0], 3, stride=1),
            nn.ELU(),
                
            nn.ConvTranspose2d(k_s[0], k_s[1], 5, stride=1),
            nn.ELU(),
                
            nn.ConvTranspose2d(k_s[1], k_s[1], 5, stride=2),
            nn.ELU(),
                
            nn.ConvTranspose2d(k_s[1], k_s[2], 3, stride=1),
            nn.ReLU(),
            
            nn.ConvTranspose2d(k_s[2], 3, 3, stride=1),
            nn.Sigmoid()
            )
        self.recon_loss = nn.BCELoss()
        

    def forward(self, x):        
        enc   = self.convEnc(x)
        encL  = enc.reshape(enc.size(0), -1)        
        muV   = self.mu(encL)
        varV  = self.var(encL)
        
        out   = self._reparameterize(muV, varV)
        
        lin2d = out.reshape([-1, 1, 8, 8])
        
        decon = self.convDec(lin2d)
        out   = decon[:, :, 1:-2, 1:-2]
    
        return out, muV, varV           
        
        print("enc", enc.shape)
        print("encL", encL.shape)
        print("out", out.shape)
        print("lin2d", lin2d.shape)
        print("decon", decon.shape) 
        print("out", out.shape) 
        print("mu", muV.shape)
        print("var", varV.shape)
    
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
        KLD /= x.shape[0] * 1024  * 3
        return BCE,  KLD
