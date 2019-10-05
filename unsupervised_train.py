#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 18:52:16 2019

@author: ishwark
"""

import torch
import torchvision as tv
import numpy as np
import time
from torchsummary import summary

from torch.utils.tensorboard import SummaryWriter

from unsupervised_model import ConvAutoEncoder
from data_loader import CifarLoader

from torch.utils.data import DataLoader
import torchvision.datasets as datasets

class TestGroup:
    def __init__(self, label, num_tests = 20):
        self.num_tests = num_tests
        self._items = []
        self.label = label

    def add_item(self, item):
        self._items.append(item)

    def evaluate(self, model):
        distance = 0
        for i in range(0, self.num_tests):
            idx1 = np.random.randint(0, len(self._items))
            idx2 = np.random.randint(0, len(self._items))
            mu1 = model(self._items[idx1])
            while idx1 == idx2:
                idx2 = np.random.randint(0, len(self._items))
            mu2 = model(self._items[idx2])
            
            
            distance = distance + torch.dot(mu1.view(-1), mu2.view(-1))
        
        return distance.item()/self.num_tests


class UnsupervisedTrainer:
    def __init__(self, 
                 exp_type   = 'CIFAR',
                 max_epochs = 100,
                 log_freq   = 50, # in batches
                 model_save = 10, 
                 test_freq  = 200):

        super(UnsupervisedTrainer, self).__init__()

        self.BATCH_SIZE = 1000
        self.LOG_FREQ   = log_freq# in batches
        self.MAX_EPOCHS = max_epochs
        self.MODEL_SAVE = model_save
        self.EXP_TYPE   = exp_type
        self.TEST_FREQ  = test_freq

        self.device = torch.device("cpu")
        if torch.cuda.device_count() >=1:
            self.device = torch.device("cuda:0")  
            
        self.float_dtype = torch.float32

        toTensor = tv.transforms.ToTensor()

        if self.EXP_TYPE == 'CIFAR':
            dataset = CifarLoader('./data')
            testset = CifarLoader('./data', True)
            im_shape = [3, 32, 32]
            rec_shape = [1, 32, 32]
        elif self.EXP_TYPE == 'MNIST-Fashion':
            dataset = datasets.FashionMNIST('./data', transform=toTensor)
            testset = datasets.FashionMNIST('./data', False, transform=toTensor)
            im_shape = [1, 28, 28]
            rec_shape = im_shape
        elif self.EXP_TYPE == 'MNIST':
            dataset = datasets.MNIST(root='./data', transform=toTensor)
            testset = datasets.MNIST('./data', False, transform=toTensor)
            im_shape = [1, 28, 28]
            rec_shape = im_shape

        self.train_loader = DataLoader(dataset, 
                                       batch_size=self.BATCH_SIZE, shuffle=True, 
                                       num_workers=2, pin_memory=True)
        
        self.model = ConvAutoEncoder(im_shape, rec_shape)
        self.model.to(self.device).to(self.float_dtype)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4, 
                                          weight_decay=1e-5)        
        self._summarize(im_shape)

        self.test_groups = []
        self._create_testdata(testset)

    def _create_testdata(self, testset):
        test_loader = DataLoader(testset, batch_size=1)

        for label in testset.classes:
            self.test_groups.append(TestGroup(str(label), 10))

        for bt, batch_label in enumerate(test_loader):
            self.test_groups[batch_label[1]].add_item(batch_label[0].to(self.device))
        
        print("Test group sizes:")
        for grp in self.test_groups:
            print(grp.label, '->' , len(grp._items))


    def _summarize(self, im_shape):

        dummy_size = tuple( [self.BATCH_SIZE] + im_shape)
        dummy_input = torch.Tensor(torch.ones(dummy_size)).to(self.device)
        self.writer = SummaryWriter('runs/auto_enc/{}'.format(self.EXP_TYPE))
        self.writer.add_graph(self.model, dummy_input)
        self.writer.flush()

        summary(self.model, tuple(im_shape))


    def test_step(self, test_step):
        self.model.train(False)
        named_loss = {}
        mean_dist = 0
        for grp in self.test_groups:
            err = grp.evaluate(self.model)
            named_loss[grp.label] = err
            mean_dist = mean_dist + err
            
        mean_dist = mean_dist/len(self.test_groups)
        self.writer.add_scalars('InstMeanDist', named_loss, test_step)
        self.writer.add_scalar('MeanDist', mean_dist, test_step)

        self.model.train(True)
        print('Mean Distance: ', mean_dist)
        


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

                if step % self.TEST_FREQ == 0:
                    print('------------------TEST-------------------')
                    self.test_step(step)
                    print('------------------TEST-------------------')
                    
            if epoch % self.MODEL_SAVE == 0:
                torch.save(self.model.state_dict(), "epoch_{}.model".format(epoch))
        
        torch.save(self.model.state_dict(), "epoch_final.model")
