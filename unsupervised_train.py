#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 18:52:16 2019

@author: ishwark
"""

import time

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
        self._items = []
        self.label = label

    def add_item(self, item):
        """Add an image item, known to be in this cluster"""
        self._items.append(item)

    def evaluate(self, model):
        distance = 0
        for _ in range(0, self.num_tests):
            idx1 = np.random.randint(0, len(self._items))
            idx2 = np.random.randint(0, len(self._items))
            mu1 = model(self._items[idx1])
            while idx1 == idx2:
                idx2 = np.random.randint(0, len(self._items))
            mu2 = model(self._items[idx2])
            mu1 /= torch.norm(mu1)
            mu2 /= torch.norm(mu2)
            distance += F.cosine_similarity(mu1, mu2)

        return distance.item()/self.num_tests


class UnsupervisedTrainer:
    """Trainign to make clusters of images"""

    def __init__(self,
                 exp_type='CIFAR',
                 max_epochs=100,
                 log_freq=50,  # in batches
                 model_save=10,
                 test_freq=200):

        super(UnsupervisedTrainer, self).__init__()

        self.batch_size = 1000
        self.log_freq = log_freq  # in batches
        self.max_epochs = max_epochs
        self.model_save_freq = model_save
        self.exp_type = exp_type
        self.test_freq = test_freq

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
            dataset = datasets.FashionMNIST('./data', transform=to_tensor)
            testset = datasets.FashionMNIST(
                './data', False, transform=to_tensor)
            im_shape = [1, 28, 28]
            rec_shape = im_shape
        elif self.exp_type == 'MNIST':
            dataset = datasets.MNIST(root='./data', transform=to_tensor)
            testset = datasets.MNIST('./data', False, transform=to_tensor)
            im_shape = [1, 28, 28]
            rec_shape = im_shape

        self.train_loader = DataLoader(dataset,
                                       batch_size=self.batch_size,
                                       shuffle=True, pin_memory=True)

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
            self.test_groups.append(TestCluster(str(label), 10))

        for _, batch_label in enumerate(test_loader):
            batch, label = batch_label[0].to(self.device), batch_label[1]
            self.test_groups[label].add_item(batch)

        print("Test group sizes:")
        for grp in self.test_groups:
            print(grp.label, '->', len(grp._items))

    def _summarize(self, im_shape):

        dummy_size = tuple([self.batch_size] + im_shape)
        dummy_input = torch.Tensor(torch.ones(dummy_size)).to(self.device)
        self.writer = SummaryWriter('runs/auto_enc/{}'.format(self.exp_type))
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
        total_steps = len(self.train_loader) * self.max_epochs

        print("Training {} steps with device set to {}".format(total_steps,
                                                               self.device))
        step = 0
        start = time.time()
        for epoch in range(0, self.max_epochs):
            for bt, batch_label in enumerate(self.train_loader):

                batch = batch_label[0].to(self.device)
                recon, mu, logvar = self.model(batch)

                progress = step / total_steps
                batch_1c = torch.mean(batch, axis=1, keepdim=True)
                bc, kl = self.model.loss_function(recon, batch_1c, mu, logvar)
                loss = bc + kl

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if step % self.log_freq == 0:
                    dur = 1000 * ((time.time() - start) /
                                  (self.batch_size * self.log_freq))
                    print("%d,\t%d:\t%1.5f = %1.5f + %1.5f | %2.1f%% | %1.3fms. | %s" %
                          (epoch, bt, loss.item(), bc.item(), kl.item(),
                           progress*100, dur, '-'*int(progress*40)))

                    self.writer.add_image(
                        'Train-Images', tv.utils.make_grid(batch[0:4]), step)
                    self.writer.add_image(
                        'Train-1C-Images', tv.utils.make_grid(batch_1c[0:4]), step)
                    self.writer.add_image(
                        'Recon-Images', tv.utils.make_grid(recon[0:4]), step)

                    self.writer.add_scalars(
                        'Loss', {'total': loss, 'bc': bc, 'kl': kl}, step)

                    start = time.time()
                step = step + 1

                if step % self.test_freq == 0:
                    print('------------------TEST-------------------')
                    self.test_step(step)
                    print('------------------TEST-------------------')

            if epoch % self.model_save_freq == 0:
                torch.save(self.model.state_dict(),
                           "epoch_{}.model".format(epoch))

        torch.save(self.model.state_dict(), "epoch_final.model")


UnsupervisedTrainer('MNIST').train_loop()
