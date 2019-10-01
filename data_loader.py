# -*- coding: utf-8 -*-
"""
Data Loader of sorts, works with CIFAR-10 dataset from 
       here: http://www.cs.toronto.edu/~kriz/cifar.html 
   
main functions: 
    _load_all_batches()
        returns 2 dictionaries of batches: 'train' and 'test', 
        These batches look like this:
            b'labels'      : List of label intergers, 0-9
            b'num_images'  : int number of images
            b'images'      : numpy.ndarray of shape (N, X, Y, C)
            b'label_names' : List of byte string describing labels
            
    get_one_image()
        given an index, returns label (an integer) and a torch 4D tensor
"""

import numpy as np
import pickle
import os
import torch

class CifarLoader:
    label_names = []
    def __init__(self, cifar_loc='./', 
                 device=torch.device("cpu"),
                 prec=torch.float32):
        
        super(CifarLoader, self).__init__()
        self.cifar_loc = cifar_loc
        self.device = device
        self.precision = prec

        self.train_batches = {}
        self.test_batch = {}        
        self._load_all_batches()

    def num_images(self, train=True):
        return self.train_batches[b'num_images'] if train else self.test_batches[b'num_images']

    def _load_batch(self, file, prev_batches=None):
        batch = {}
        with open(file, 'rb') as fo:
            batch = pickle.load(fo, encoding='bytes')
           
        num_instances = len(batch[b'filenames'])
        batch[b'num_images'] = num_instances
        images = np.reshape(batch[b'data'], [num_instances, 3, 32, 32])
        batch[b'images'] = images.astype(np.float16)/256
           
        del batch[b'data']
        del batch[b'filenames']
        del batch[b'batch_label']
        
        if prev_batches:
            batch[b'num_images'] =  num_instances + prev_batches[b'num_images']
            batch[b'images'] = np.concatenate((batch[b'images'], prev_batches[b'images']), axis=0)
            batch[b'labels'] = batch[b'labels'] + prev_batches[b'labels']
            
        return batch
    
    def _load_all_batches(self):
        
        train_batches = None
        for i in range(1,6):
            name = '{}/cifar-10-batches-py/data_batch_{}'.format(self.cifar_loc,i)
            train_batches = self._load_batch(name,train_batches)
        
        self.train_batches = train_batches
        name = '{}/cifar-10-batches-py/test_batch'.format(self.cifar_loc,)
        self.test_batch = self._load_batch(name)
        
        meta_fn = os.path.join(self.cifar_loc, 'cifar-10-batches-py/batches.meta')
        with open(meta_fn, 'rb') as fo:
            meta = pickle.load(fo, encoding='bytes')
            CifarLoader.label_names = meta[b'label_names']

    def shuffle(self, train=True):
        if train:
            np.random.shuffle(self.train_batches[b'images'])
        else: 
            np.random.shuffle(self.test_batch [b'images'])
    
    def get_one_image_batch(self, idx, train=True, batch_size=1):
        
        batches = self.train_batches if train else self.test_batches
        label = batches[b'labels'][idx:idx+batch_size]
        npimg = batches[b'images'][idx:idx+batch_size]
        img = torch.from_numpy(npimg).to(self.device).to(self.precision)
        return label, img
    
    @staticmethod
    def display_img(img, label):
        import matplotlib as plt
        import numpy as np
        img = img.numpy()
        img = np.transpose(img, [1, 2, 0])
        lstr = "{} : {}".format(label, label)
        plt.pyplot.title(lstr)
        plt.pyplot.imshow(img, interpolation='bicubic')
        
    
