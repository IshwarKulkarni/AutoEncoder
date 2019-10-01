#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 18:52:16 2019

@author: ishwark
"""

#%%

from unsupervised_model import ConvAutoEncoder
from data_loader import CifarLoader
import torch

BATCH_SIZE = 125
PRINT_FREQ = 20 # in batches
PLOT_FREQ  = 125 # in batches

device = torch.device("cuda:0")  if True else torch.device("cpu")
float_dtype = torch.float32

model = ConvAutoEncoder()
model.to(device).to(float_dtype)

dl = CifarLoader('/home/ishwark/autoencoder/', device, float_dtype)


#%%


import time
import matplotlib.pyplot as plt
from torch import optim

optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)        
model.train()

num_batches = dl.num_images() // BATCH_SIZE
 
losses = []
for epoch in range(0, 25):
    dl.shuffle()
    for bt in range(0, num_batches):
        start = time.time()
        
        _, batch = dl.get_one_image_batch(bt * BATCH_SIZE, True, BATCH_SIZE) 
                
        recon, mu, logvar = model(batch)
        bc, kl = model.loss_function(recon, batch, mu, logvar)
        loss = bc + kl
        
        losses.append(loss.item())
        
        optimizer.zero_grad() 
        loss.backward()        
        optimizer.step()        
        
        
        if bt % PRINT_FREQ == 0:
            print("%d:%d\t Loss:%1.4f, BCE:%1.4f, KL:%1.4f, \t%1.3fs, %d"%(
                    epoch, bt,
                    loss.item() , bc.item(), kl.item(), time.time()-start, 
                    bt*BATCH_SIZE))

        if bt % PLOT_FREQ == 0:
            
            fig = plt.figure()
            grph = plt.plot( range(len(losses)), losses)
                                    
            fig = plt.figure()
            a = fig.add_subplot(1, 2, 1)
            CifarLoader.display_img(batch[0].cpu().detach(), "img")
            a.set_title('Image: {}'.format(bt * BATCH_SIZE))
            a = fig.add_subplot(1, 2, 2)
            CifarLoader.display_img(recon[0].cpu().detach(), "recon")
            a.set_title('Recon')
            
            plt.show()
        
        
