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
from torch.utils.data import Dataset

class CifarLoader(Dataset):
    label_names = []
    def __init__(self, cifar_loc='./', is_test=False):
        
        super(CifarLoader, self).__init__()
        self.cifar_loc = cifar_loc
        self._num_images = 0
        self.batches = {}
        self._load_all_batches()

    def _load_batch(self, file, prev_batches=None):
        batch = {}
        with open(file, 'rb') as fo:
            batch = pickle.load(fo, encoding='bytes')
           
        num_instances = len(batch[b'filenames'])
        self._num_images += num_instances
        batch[b'num_images'] = num_instances
        batch[b'images'] = batch[b'data'] 
           
        del batch[b'data']
        del batch[b'filenames']
        del batch[b'batch_label']
        
        if prev_batches:
            batch[b'images'] = np.concatenate((batch[b'images'], prev_batches[b'images']), axis=0)
            batch[b'labels'] = batch[b'labels'] + prev_batches[b'labels']
            
        return batch
    
    def _load_all_batches(self, is_test=False):
        
        if is_test:
            name = '{}/cifar-10-batches-py/test_batch'.format(self.cifar_loc,)
            self.batches = self._load_batch(name)
        else:
            batches = None
            for i in range(1,6):
                name = '{}/cifar-10-batches-py/data_batch_{}'.format(self.cifar_loc,i)
                batches = self._load_batch(name,batches)
                self.batches = batches
        
        meta_fn = os.path.join(self.cifar_loc, 'cifar-10-batches-py/batches.meta')
        with open(meta_fn, 'rb') as fo:
            meta = pickle.load(fo, encoding='bytes')
            CifarLoader.label_names = meta[b'label_names']

    def __len__(self):
        return self._num_images

    
    def __getitem__(self, idx):
        label = self.batches[b'labels'][idx]
        image = self.batches[b'images'][idx]
        image = np.reshape(image, (3, 32, 32))
        image = image.astype(np.float32) / 256.0
        return image, label
    
    @staticmethod
    def display_img(img, label):
        import matplotlib as plt
        import numpy as np
        img = img.numpy()
        img = np.transpose(img, [1, 2, 0])
        lstr = "{} : {}".format(label, label)
        plt.pyplot.title(lstr)
        plt.pyplot.imshow(img, interpolation='bicubic')
        
    
