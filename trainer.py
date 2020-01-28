#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 18:52:16 2019

@author: ishwark
"""

import time
import datetime

import os

import torch
import math
import torchvision as tv
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

from autoencoder_model import ConvAutoEncoder
from test import Tester

# pylint: disable=E1101
# pylint: disable=R0902
# pylint: disable=R0914

class UnsupervisedTrainer:
    """Training to make clusters of images"""

    def __init__(self, exp_type='CIFAR'):

        super(UnsupervisedTrainer, self).__init__()

        self.batch_size = 256
        self.log_freq = 200  # in steps
        self.max_epochs = 125
        self.exp_type = exp_type
        self.test_step_freq = 200

        self.embedding_size = 16

        self.c_size = 10

        self.artifacts_dir = 'runs/{}'.format(self.exp_type)
        to_tensor = tv.transforms.ToTensor()

        if self.exp_type == 'CIFAR':
            dataset = datasets.CIFAR10('./data', transform=to_tensor, download=True)
            testset = datasets.CIFAR10('./data', transform=to_tensor, download=True, train=False)
            im_shape = [3, 32, 32]
            rec_shape = [1, 32, 32]
        elif self.exp_type == 'MNIST-Fashion':
            dataset = datasets.FashionMNIST('./data', transform=to_tensor, download=True)
            testset = datasets.FashionMNIST('./data', transform=to_tensor, download=True,
                                            train=False)
            im_shape = [1, 28, 28]
            rec_shape = im_shape
        elif self.exp_type == 'MNIST':
            dataset = datasets.MNIST('./data', transform=to_tensor, download=True)
            testset = datasets.MNIST('./data', transform=to_tensor, download=True,
                                     train=False)
            im_shape = [1, 28, 28]
            rec_shape = im_shape

        self.train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True,
                                       pin_memory=True)

        print("Trainset size: {}, testset size:{} ".format(len(dataset), len(testset)))

        self.model = ConvAutoEncoder(im_shape, rec_shape, z_size=self.embedding_size, c_size=self.c_size)
        try:
            model_file = os.path.join(self.artifacts_dir, 'final_epoch.model')
            print(f"Looking for model to load at {model_file}", end=' ')
            self.model.load_state_dict(torch.load(model_file))
            print(".. Done!")
        except:
            print(".. not found!")

        self.device = torch.device("cpu")
        if torch.cuda.device_count() > 0:
            self.device = torch.device(0)

        self.model.train(True).to(self.device)

        self.writer = SummaryWriter(self.artifacts_dir)
        self._summarize(im_shape)

        if torch.cuda.device_count() >= 1:
            self.device = torch.device(1)
            self.model.train(True).to(self.device)

        init_lr = 5e-5
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=init_lr, weight_decay=1e-5)
        self.lr_sched = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.97)

        self.tester = Tester(testset, self.writer, self.device, self.embedding_size, self.c_size)

    def _summarize(self, im_shape):

        dummy_size = tuple([self.batch_size] + im_shape)
        self.model.train(True)
        self.model.fake_training = True
        dummy_ip = [torch.ones(dummy_size, device=self.device),
                    torch.zeros([self.batch_size, self.c_size], device=self.device)]
        self.writer.add_graph(self.model, dummy_ip)
        self.model.fake_training = False
        self.writer.flush()

        if self.c_size is 0:
            summary(self.model, tuple(im_shape))
        else:
            num_params = sum(p.numel() for p in self.model.parameters())
            print(f"Number of params: {num_params}")

    def train_loop(self):
        """Runs MAX_EPOCHS number of epochs"""
        self.model.train()
        total_steps = len(self.train_loader) * self.max_epochs

        print(f"Training {total_steps} steps @ {len(self.train_loader)} steps/epoch "
              f"with device set to \"{self.device}\" ")

        label_1hot = None
        if self.c_size > 0:
            label_1hot = torch.FloatTensor(self.batch_size, self.c_size).to(self.device)
        step = 0
        start = time.time()
        
        for epoch in range(0, self.max_epochs):
            for batch_label in self.train_loader:

                if batch_label[0].shape[0] != self.batch_size:
                    # some batches are smaller (like the last one)
                    continue

                batch = batch_label[0].to(self.device)
                
                if label_1hot is not None:
                    label = batch_label[1].unsqueeze_(1).to(self.device)
                    label_1hot.zero_()
                    label_1hot.scatter_(1, label, value=1)

                recon, mean, logvar = self.model(batch, label_1hot)

                progress = step / total_steps
                batch_1c = torch.mean(batch, axis=1, keepdim=True)
                bc_loss, kl_loss = self.model.loss_function(recon, batch_1c, mean, logvar)
                scale = math.sin(progress *  1.57079632679) # 0 .. pi/2
                bc_scale = scale * scale
                kl_scale = 1 - bc_scale
                loss = bc_loss * bc_scale + kl_loss * kl_scale

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if step % self.log_freq == 0:
                    dur = 1000 * ((time.time() - start) /
                                  (self.batch_size * self.log_freq))
                    print(datetime.datetime.now().strftime('%m/%d %H:%M:%S'), end=': ')
                    print("Step %04d, %1.5f = %1.5f + %1.5f | %02.2f%% | %1.3fms." %
                          (step, loss.item(), bc_loss.item(), kl_loss.item(), progress*100, dur))

                    self.writer.add_image('Train/Images', tv.utils.make_grid(batch[0:4]), step)
                    self.writer.add_image('Train/Recon', tv.utils.make_grid(recon[0:4]), step)
                    self.writer.add_scalar('LR', self.optimizer.param_groups[0]['lr'], step)

                    losses = {'Total': loss, 'bc': bc_loss, 'kl': kl_loss} 
                    #, 'bc_scale' : bc_scale, 'kl_scale' : kl_scale}
                    self.writer.add_scalars('Losses', losses, step)

                    start = time.time()

                if step % self.test_step_freq == 0:
                    print(f'------------------TEST-{step}-------------------')
                    self.save_model(f"epoch_{step}.model")
                    self.model.train(False)
                    self.tester.step(self.model, step)
                    self.model.train(True)
                    print(f'------------------TEST-{step}-------------------')

                step = step + 1

            if epoch >= self.max_epochs/10:
                self.lr_sched.step()

        self.save_model("final_epoch.model")

    def save_model(self, filename):
        """ Saving the model at artifacts directory """
        outfile = os.path.join(self.artifacts_dir, filename)
        torch.save(self.model.state_dict(), outfile)
