#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 18:52:16 2019

@author: ishwark
"""

import torch
import torchvision as tv

from torch.utils.tensorboard import SummaryWriter

from unsupervised_model import ConvAutoEncoder
from data_loader import CifarLoader

from torch.utils.data import DataLoader

BATCH_SIZE = 1000
LOG_FREQ   = 50 # in batches

device = torch.device("cuda:0")  if torch.cuda.device_count() >=1 else torch.device("cpu")
float_dtype = torch.float32

model = ConvAutoEncoder()
model.to(device).to(float_dtype)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)        

train_loader = DataLoader(CifarLoader(),
                          batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
test_loader  = DataLoader(CifarLoader(is_test=True), num_workers=2)

with SummaryWriter('runs/auto_enc') as writer:
    writer.add_graph(model,  torch.Tensor(torch.rand(BATCH_SIZE, 3, 32, 32)).to(device))
    writer.flush()
# %%

model.train()
step = 0
for epoch in range(0, 100):
    for bt, batch_label in enumerate(train_loader):
        batch = batch_label[0].to(device)
        recon, mu, logvar = model(batch)
        
        batch_1c = torch.mean(batch, axis=1, keepdim = True)        
        bc, kl = model.loss_function(recon, batch_1c, mu, logvar)
        loss = bc + kl

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % LOG_FREQ == 0:
            print("%d,\t%d:\t%1.5f = %1.5f + %1.5f"%(epoch, step, loss.item(),  bc.item(), kl.item()))
            writer.add_image('Train-Images', tv.utils.make_grid(batch[0:4]), step)
            writer.add_image('Train-1C-Images', tv.utils.make_grid(batch_1c[0:4]), step)
            writer.add_image('Recon-Images', tv.utils.make_grid(recon[0:4]), step)
            
            writer.add_scalars('Loss', {'total':loss, 'bc':bc, 'kl': kl}, step)
            
            
        step = step + 1
    torch.save(model.state_dict(), "epoch_{}.model".format(epoch))
            
            
        
