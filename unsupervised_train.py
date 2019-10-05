#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 18:52:16 2019

@author: ishwark
"""

import torch
import torchvision as tv
import time
from torchsummary import summary

from torch.utils.tensorboard import SummaryWriter

from unsupervised_model import ConvAutoEncoder
from data_loader import CifarLoader

from torch.utils.data import DataLoader
import torchvision.datasets as datasets


    
class UnsupervisedTrainer:
    
    def __init__(self, 
                 exp_type   = 'CIFAR',
                 max_epochs = 100,
                 log_freq   = 50, # in batches
                 model_save = 10):
        
        super(UnsupervisedTrainer, self).__init__()
    
        self.BATCH_SIZE = 1000
        self.LOG_FREQ   = log_freq# in batches
        self.MAX_EPOCHS = max_epochs
        self.MODEL_SAVE = model_save
        self.EXP_TYPE   = exp_type
    
        self.device = torch.device("cuda:0")  if torch.cuda.device_count() >=1 else torch.device("cpu")
        self.float_dtype = torch.float32
        
        if self.EXP_TYPE == 'CIFAR':
            dataset = CifarLoader('./data')
            im_shape = [3, 32, 32]
            rec_shape = [1, 32, 32]
        elif self.EXP_TYPE == 'MNIST-Fashion':
            dataset = datasets.FashionMNIST('./data', transform=tv.transforms.ToTensor())
            im_shape = [1, 28, 28]
            rec_shape = im_shape
        elif self.EXP_TYPE == 'MNIST':
            dataset = datasets.MNIST(root='./data', transform=tv.transforms.ToTensor())
            im_shape = [1, 28, 28]
            rec_shape = im_shape
        
        self.train_loader = DataLoader(dataset, 
                                       batch_size=self.BATCH_SIZE, shuffle=True, 
                                       num_workers=2, pin_memory=True)
        
        self.model = ConvAutoEncoder(im_shape, rec_shape)
        self.model.to(self.device).to(self.float_dtype)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=1e-5)        
        self._summarize(im_shape)
        
        self._create_testdata()
        
    def _create_testdata(self):
        pass
        
    def _summarize(self, im_shape):
        dummy_size = tuple( [self.BATCH_SIZE] + im_shape)
        dummy_input = torch.Tensor(torch.ones(dummy_size)).to(self.device)
        with SummaryWriter('runs/auto_enc/{}'.format(self.EXP_TYPE)) as self.writer:
            self.writer.add_graph(self.model, dummy_input)
            self.writer.flush()
            
        summary(self.model, tuple(im_shape))
    
    # %%
    def test_step(self):
        self.model.train(False)
        
        self.model.train(True)
        
    
    # %%
    def train_loop(self):
        self.model.train()
        total_steps = len(self.train_loader) * self.MAX_EPOCHS
        
        print("Training {} steps with device set to {}".format(total_steps, 
                                                               self.device))
        step = 0
        start = time.time()
        for epoch in range(0, self.MAX_EPOCHS):
            for bt, batch_label in enumerate(self.train_loader):
                
                batch = batch_label[0].to(self.device)
                recon, mu, logvar = self.model(batch)
                
                progress = step / total_steps
                batch_1c = torch.mean(batch, axis=1, keepdim = True)        
                bc, kl = self.model.loss_function(recon, batch_1c, mu, logvar)
                loss = bc + kl * progress * 2
        
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        
                if step % self.LOG_FREQ == 0:
                    dur = 1000 * ( (time.time() - start)/ (self.BATCH_SIZE * self.LOG_FREQ) ) 
                    print("%d,\t%d:\t%1.5f = %1.5f + %1.5f | %2.1f%% | %1.3fms. | %s"%
                          (epoch, bt, loss.item(),  bc.item(), kl.item(), 
                           progress*100, dur, '-'*int(progress*40)))
                    
                    self.writer.add_image('Train-Images', tv.utils.make_grid(batch[0:4]), step)
                    self.writer.add_image('Train-1C-Images', tv.utils.make_grid(batch_1c[0:4]), step)
                    self.writer.add_image('Recon-Images', tv.utils.make_grid(recon[0:4]), step)
                    
                    self.writer.add_scalars('Loss', {'total':loss, 'bc':bc, 'kl': kl}, step)
                    
                    start = time.time()
                step = step + 1

            self.test_step()
            if epoch % self.MODEL_SAVE == 0:
                torch.save(self.model.state_dict(), "epoch_{}.model".format(epoch))
        
        torch.save(self.model.state_dict(), "epoch_final.model")

ut = UnsupervisedTrainer('MNIST')
ut.train_loop()