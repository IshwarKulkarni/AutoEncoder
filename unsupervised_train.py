#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 18:52:16 2019

@author: ishwark
"""

import time

import random
import numpy as np
import torch
import torchvision as tv
import torchvision.datasets as datasets
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

from data_set import CifarDataset
from unsupervised_model import ConvAutoEncoder

# pylint: disable=too-many-instance-attributes


class TestCluster:
    """A known test cluster"""

    def __init__(self, label, num_tests=20):
        self.num_tests = num_tests
        self._batches = []
        self.label = label
        self.batch_mu = None

    def add_item(self, item):
        """Add an image item, known to be in this cluster"""
        self._batches.append(item)

    def rebatch(self, batch_size):
        assert len(self._batches) > batch_size, "Need atleast {} items to batch".format(batch_size)
        batches = []
        for b in range(0, len(self._batches), batch_size):
            batch = self._batches[b:b+batch_size]
            batch = torch.squeeze(torch.stack(batch), 1)
            batches.append(batch)
        self._batches = batches
        self.metadata = [self.label] * batch_size

    def intra_cluster_std(self, model):
        r = random.randint(0, len(self._batches)-2)
        batch_mu = model(self._batches[r])
        self.batch_mu = batch_mu
        self.center = batch_mu.mean(dim=0)
        return self.batch_mu.std(dim = 0).mean()

    def intra_cluster_sim(self, model, num_samples =10):
        r = random.randint(0, len(self._batches)-2)
        batch_mu = model(self._batches[r])
        self.batch_mu = batch_mu
        self.center = batch_mu.mean(dim=0)
        n = batch_mu.shape[0] - 1
        mean_sim = 0
        for _ in range(0, num_samples):
            r1 = random.randint(0, n)
            r2 = random.randint(0, n)
            while r1 == r2:
                r2 = random.randint(0, n)
            mean_sim += F.cosine_similarity(batch_mu[r1], batch_mu[r2], dim=0)

        return mean_sim / num_samples

    
class UnsupervisedTrainer:
    """Training to make clusters of images"""

    def __init__(self,
                 exp_type='CIFAR',
                 max_epochs=150,
                 log_freq=50,  # in batches
                 model_save_epoch=12):

        super(UnsupervisedTrainer, self).__init__()

        self.batch_size = 1000
        self.log_freq = log_freq  # in batches
        self.max_epochs = max_epochs
        self.model_save_freq_epoch = model_save_epoch
        self.exp_type = exp_type
        self.test_step_freq = 100
        
        self.device = torch.device("cpu")
        if torch.cuda.device_count() >= 1:
            self.device = torch.device("cuda:0")

        self.float_dtype = torch.float32

        to_tensor = tv.transforms.ToTensor()

        if self.exp_type == 'CIFAR':
            dataset = CifarDataset('./data')
            testset = CifarDataset('./data', True)
            im_shape = [3, 32, 32]
            rec_shape = [1, 32, 32]
        elif self.exp_type == 'MNIST-Fashion':
            dataset = datasets.FashionMNIST('./data', transform=to_tensor, download=True)
            testset = datasets.FashionMNIST(
                './data', False, transform=to_tensor, download=True)
            im_shape = [1, 28, 28]
            rec_shape = im_shape
        elif self.exp_type == 'MNIST':
            dataset = datasets.MNIST(root='./data', transform=to_tensor, download=True)
            testset = datasets.MNIST('./data', False, transform=to_tensor, download=True)
            im_shape = [1, 28, 28]
            rec_shape = im_shape

        self.train_loader = DataLoader(dataset,
                                       batch_size=self.batch_size,
                                       shuffle=True, pin_memory=True)

        print("Trainset size: {}, testset size:{} ".format(len(dataset), len(testset)))

        self.model = ConvAutoEncoder(im_shape, rec_shape)
        try:
            self.model.load_state_dict(torch.load('final_epoch.model'))
        except:
            pass
        self.model.train(True).to(self.device)

        init_lr = 5e-4
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=init_lr, weight_decay=1e-5)
        self.lr_sched  = torch.optim.lr_scheduler.ExponentialLR(self.optimizer,
                                                           gamma=0.95, last_epoch=-1)

        self._summarize(im_shape)

        self.model.to(self.float_dtype)

        self.test_groups = []
        self._create_testdata(testset)

    def _create_testdata(self, testset):
        test_loader = DataLoader(testset, batch_size=1)

        for label in testset.classes:
            self.test_groups.append(TestCluster(str(label), 10))

        for _, batch_label in enumerate(test_loader):
            batch, label = batch_label[0].to(self.device), batch_label[1]
            self.test_groups[label].add_item(batch)

        for grp in self.test_groups:
            grp.rebatch(16)

        print("Test group sizes:")
        for grp in self.test_groups:
            print(grp.label, '->', len(grp._batches))

    def _summarize(self, im_shape):

        dummy_size = tuple([self.batch_size] + im_shape)
        self.model.train(True)
        self.writer = SummaryWriter('runs//{}'.format(self.exp_type))
        self.model.fake_training = True
        self.writer.add_graph(self.model, torch.ones(dummy_size).to(self.device))
        self.model.fake_training = False
        self.writer.flush()  
        summary(self.model, tuple(im_shape))


    def test_step(self, test_step):
        self.model.train(False)
        named_loss = {}
        total_similarity = 0
        centers = []
        batch_mus = []
        meta = []
        for grp in self.test_groups:
            std = grp.intra_cluster_sim(self.model)
            named_loss[grp.label] = std
            total_similarity = total_similarity + std
            centers.append(grp.center)
            batch_mus.append(grp.batch_mu[:32])
            meta += grp.metadata[:32]

        batch_mus = torch.cat(batch_mus, 0)
        centers = torch.stack(centers)
        inter_cluster_std = centers.std(dim=0).mean()

        self.writer.add_scalars('Intra Cluster Similarity/Clusters', named_loss, test_step)
        self.writer.add_scalar('Intra Cluster Similarity/Total', total_similarity, test_step)
        self.writer.add_scalar('Inter Cluster std.', inter_cluster_std, test_step)
        self.writer.add_embedding(batch_mus, global_step=test_step,metadata=meta)
        
        self.model.train(True)
        print('Mean Inter Cluster Dist.: {} / Step: {}'.format(inter_cluster_std.item(), test_step))

    def train_loop(self):
        self.model.train()
        total_steps = len(self.train_loader) * self.max_epochs

        print("Training {} steps @ {}steps/epoch with device set to {}".format(
                total_steps, len(self.train_loader), self.device))
        step = 0
        start = time.time()
        for epoch in range(0, self.max_epochs):
            for bt, batch_label in enumerate(self.train_loader):

                batch = batch_label[0].to(self.device).to(self.float_dtype)
                recon, mu, logvar = self.model(batch)

                progress = step / total_steps
                batch_1c = torch.mean(batch, axis=1, keepdim=True)
                bc, kl = self.model.loss_function(recon, batch_1c, mu, logvar)
                loss = bc + kl

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                step = step + 1                

                if step % self.log_freq == 0:
                    dur = 1000 * ((time.time() - start) /
                                  (self.batch_size * self.log_freq))
                    print("%d,\t%d:\t%1.5f = %1.5f + %1.5f | %2.1f%% | %1.3fms. | %s" %
                          (epoch, bt, loss.item(), bc.item(), kl.item(),
                           progress*100, dur, '-'*int(progress*40)))

                    self.writer.add_image(
                        'Train-Images', tv.utils.make_grid(batch[0:4]), step)
                    #self.writer.add_image('Train-1C-Images', tv.utils.make_grid(batch_1c[0:4]), step)
                    self.writer.add_image(
                        'Recon-Images', tv.utils.make_grid(recon[0:4]), step)

                    self.writer.add_scalars(
                        'Loss', {'total': loss, 'bc': bc, 'kl': kl}, step)

                    self.writer.add_scalars( 'Optim', {
                        'LR': self.optimizer.param_groups[0]['lr'], 
                        'WD': self.optimizer.param_groups[0]['weight_decay'] }, step)

                    start = time.time()
                
                if step % self.test_step_freq == 0:
                    print('------------------TEST-------------------')
                    self.test_step(step)
                    print('------------------TEST-------------------')
            
            if epoch >= self.max_epochs/10:
                self.lr_sched.step()
            if epoch % self.model_save_freq_epoch == 0:
                torch.save(self.model.state_dict(), "epoch_{}.model".format(epoch))

        torch.save(self.model.state_dict(), "epoch_final.model")

    def play(self, name):
        trainer.model.train(False)
        self.model.load_state_dict(torch.load(name))
        for grp in self.test_groups:
            grp.intra_cluster_std(self.model)
            print( (torch.sum(grp.batch_mu, dim=0) / grp.batch_mu.shape[0]).cpu().detach().numpy() )

trainer = UnsupervisedTrainer('MNIST')
trainer.train_loop()
#trainer.play('final_epoch.model')

