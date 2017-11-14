#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import numpy as np
import scipy.misc
import h5py
import matplotlib.pyplot as plt
from data_augmentation import Augmentator
np.random.seed(123)

def myprint(s):
    fw = open('output.txt','a+')
    fw.write(s+'\n')
    fw.close()
    print(s)


# Loading data from disk
class DataLoaderDisk(object):
    def __init__(self, **kwargs):

        self.load_size = int(kwargs['load_size'])
        self.fine_size = int(kwargs['fine_size'])
        self.data_mean = np.array(kwargs['data_mean'])
        self.randomize = kwargs['randomize']
        self.data_root = os.path.join(kwargs['data_root'])
        self.Augmentator = Augmentator()

        # read data info from lists
        self.list_im = []
        self.list_lab = []
        with open(kwargs['data_list'], 'r') as f:
            for line in f:
                path, lab =line.rstrip().split(' ')
                self.list_im.append(os.path.join(self.data_root, path))
                self.list_lab.append(int(lab))
        self.list_im = np.array(self.list_im, np.object)
        self.list_lab = np.array(self.list_lab, np.int64)
        self.num = self.list_im.shape[0]
        myprint('# Images found: '+str(self.num))
        # permutation
        if self.randomize:
            perm = np.random.permutation(self.num) 
            self.list_im[:, ...] = self.list_im[perm, ...]
            self.list_lab[:] = self.list_lab[perm, ...]
        self._idx = 0
        
    def next_batch(self, batch_size):
        images_batch = np.zeros((batch_size, self.fine_size, self.fine_size, 3)) 
        labels_batch = np.zeros(batch_size)
        for i in range(batch_size):
            if self._idx < len(self.list_im):
                image = scipy.misc.imread(self.list_im[self._idx])
                image = scipy.misc.imresize(image, (self.load_size, self.load_size))
                #plt.imshow(np.uint8(image))
                #plt.show()
                #raw_input('he')
                #########################
                ### Data Augmentation ###
                #########################
                if self.randomize:
                    ### perform augmentation
                    image = self.Augmentator.seq.augment_image(image)
                ### Standardize ###
                image = image.astype(np.float32)/255.
                image = image - self.data_mean
                images_batch[i, ...] =  image
                labels_batch[i, ...] = self.list_lab[self._idx]
                #plt.imshow(np.uint8(images_batch[i, ...]))
                #plt.show()
                #raw_input('he')
                ### reset if done with entire batch
                self._idx += 1
                if self._idx == self.num:
                    self._idx = 0
        return images_batch, labels_batch
    
    def size(self):
        return self.num

    def reset(self):
        self._idx = 0




