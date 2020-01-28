#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: ishwark
"""

import random

from torch.utils.data import DataLoader
import torch
from torch.nn import functional as F
import torchvision as tv

# pylint: disable=E1101
# pylint: disable=R0903

class Tester:
    """Simple testing for C/VAE models"""
    class TestGroup:
        """Each Groupo corresponds to one classification class"""
        def __init__(self, label):
            self.label = label
            self.label_code = None
            self.embedding = None
            self.std = None
            self.batches = []
            self.metadata = []

    def __init__(self, testset, writer, device, embedding_size, c_size=0):
        self.dataloader = DataLoader(testset, batch_size=1)
        self.writer = writer
        self.embedding_size = embedding_size

        self.batch_size = 64
        self.proj_size = 16
        self.meta = []

        self.fake_c = None if c_size == 0 else torch.ones([self.batch_size, c_size]) * 0.1

        self.test_groups = []
        for label in testset.classes:
            grp = Tester.TestGroup(label)
            self.test_groups.append(grp)

        for batch_label in self.dataloader:
            batch, label = batch_label[0].to(device), batch_label[1]
            self.test_groups[label].batches.append(torch.squeeze(batch, 1))

        for tgrp in self.test_groups:
            batches = []
            for idx in range(0, len(tgrp.batches), self.batch_size):
                batch = torch.stack(tgrp.batches[idx:idx+self.batch_size])
                batches.append(batch)
            tgrp.batches = batches
            self.meta += [tgrp.label] * self.proj_size

    def step(self, model, step):
        """Do one step of the test"""
        batch_mus = []
        for tgrp in self.test_groups:
            rnd = random.randint(0, len(tgrp.batches)-2)
            conv_enc, batch_mu = model.conv_encode(tgrp.batches[rnd], self.fake_c)
            tgrp.embedding = torch.mean(batch_mu, 0)
            tgrp.std = torch.std(batch_mu)
            batch_mus.append(batch_mu[:self.proj_size])

            conv_enc = conv_enc[0:6, :, :, :]
            self.writer.add_image(f'EncodedIm-{tgrp.label}', tv.utils.make_grid(conv_enc), step)

        named_loss = {}
        for tgrp in self.test_groups:
            named_loss[tgrp.label] = tgrp.std

        batch_mus = torch.cat(batch_mus, 0)
        self.writer.add_embedding(batch_mus, global_step=step, metadata=self.meta)

        self.writer.add_scalars('IntraCluster_StdDev', named_loss, step)

        distance = 0
        cos_sim = 0
        num_grps = len(self.test_groups)
        for i in range(0, num_grps):
            for j in range(i + 1, num_grps):
                tgi = self.test_groups[i]
                tgj = self.test_groups[j]
                distance = distance + torch.dist(tgi.embedding, tgj.embedding)
                cos_sim = cos_sim + F.cosine_similarity(tgi.embedding, tgj.embedding, dim=0)

        # distance = distance /  (num_grps - 1) * num_grps/2
        self.writer.add_scalars('InterCluster', {'Dist' : distance, 'CosSim' : cos_sim}, step)
