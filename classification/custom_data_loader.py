"""
 * @author [Zizhao]
 * @email [zizhao@cise.ufl.edu]
 * @date 2017-03-02 09:24:22
 * @desc [description]
"""

import json
import os
import string
import numpy as np
import h5py
import random
import scipy.misc as misc
import glob
import pickle
import nltk
from torch.multiprocessing import Pool
from imgaug import augmenters as iaa
import torchvision.transforms as transforms
from PIL import Image

def pack_data(outputs):
    batch_imgs = []
    batch_labels = []
    batch_names = []

    for (img, label, name) in outputs:
        batch_imgs.append(img)
        batch_labels.append(label)
        batch_names.append(name)

    batch_labels = np.array(batch_labels)
    batch_imgs = np.array(batch_imgs)

    return batch_imgs, batch_labels 

class ParallDataWraper():
    def __init__(self, loader, batch_size, thread=0):
        self.loader = loader
        assert loader.shuffle == False, 'Shuffle in loader should be False'
        self.batch_size = batch_size
        self.pool = Pool(thread)
        self.create_loader(self.loader.num_data)

    def create_loader(self, num):
        # print ('--> remap and shuffle iterator')
        ids = [a for a in range(num)]
        random.shuffle(ids)
        self.targets = self.pool.imap(self.loader.next, (id for id in ids))

    def reset(self):
        self.create_loader(self.loader.num_data)

    def __next__(self):
        all_outputs = []
        for i in range(self.batch_size):
            try:
                outputs = self.targets.__next__()
            except StopIteration:
                self.create_loader(self.loader.num_data)
                outputs = self.targets.__next__()
            all_outputs.append(outputs)
        
        return pack_data(all_outputs)

def _list_valid_filenames_in_directory(directory, class_indices, 
                            white_list_formats={'png', 'jpg', 'jpeg', 'bmp'},
                            follow_links=False):

    def _recursive_list(subpath):
        return sorted(
            os.walk(subpath, followlinks=follow_links), key=lambda tpl: tpl[0])

    classes = []
    filenames = []
    subdir = os.path.basename(directory)
    basedir = os.path.dirname(directory)
    for root, _, files in _recursive_list(directory):
        # for fname in files:
        for fname in sorted(files): # zzz
            is_valid = False
            for extension in white_list_formats:
                if fname.lower().endswith('.' + extension):
                    is_valid = True
                    break
            if is_valid:
                classes.append(class_indices[subdir])
                # add filename relative to directory
                absolute_path = os.path.join(root, fname)
                # filenames.append(os.path.relpath(absolute_path, basedir))
                filenames.append(absolute_path)
            
    return classes, filenames

class DataLoader():

    def __init__(self, dataset_dir, batch_size, which_set, image_size=256, shuffle=True, use_extra_data=False):
        print('init topic data_loader ...')
        
        self.img_size = image_size
        dataset_dir = os.path.join(dataset_dir, which_set+'/img')

        if which_set == 'train':
            self.transform = transforms.Compose([ 
                        transforms.RandomRotation(degrees=5),
                        transforms.Resize(299),
                        transforms.RandomCrop(image_size),
                        transforms.RandomVerticalFlip(),
                        transforms.RandomHorizontalFlip()
                         ])
        else:
            self.transform = transforms.Compose([ 
                        transforms.Resize(299),
                        transforms.CenterCrop(image_size),
                                    ])

        classes = []
        for subdir in sorted(os.listdir(dataset_dir)):
            if os.path.isdir(os.path.join(dataset_dir, subdir)):
                classes.append(subdir)
                self.num_class = len(classes)
                class_indices = dict(zip(classes, range(len(classes))))
        self.classes, self.filenames = [], []
        for c in classes:
            c, f = _list_valid_filenames_in_directory(os.path.join(dataset_dir, c), class_indices=class_indices)
            self.classes.extend(c)
            self.filenames.extend(f)
        if use_extra_data:
            print ('\t use extra data from cls_data2')
            use_extra_dir = dataset_dir.replace('cls_data1', 'cls_data2')
            for c in classes:
                c, f = _list_valid_filenames_in_directory(os.path.join(use_extra_dir, c), class_indices=class_indices)
                self.classes.extend(c)
                self.filenames.extend(f)

        self.num_data = len(self.filenames)
        # setup iterator
        self.index = 0
        self.order = [a for a in range(0, self.num_data)]
        self.shuffle = shuffle

        self.shuffle_order()
        
        print('\t {} samples from the {} set in {} classes'.format(len(self.order), which_set, len(class_indices)))


    def shuffle_order(self):
        if self.shuffle:
            random.shuffle(self.order)
            print ('--> shuffling data ...')
        self.index = 0
    
    def next(self, idx):
        id = self.order[idx]

        local_pathname = self.filenames[id]
        name = os.path.basename(local_pathname)
        label = self.classes[id]

        raw_img = Image.open(local_pathname)
        img = self.transform(raw_img)
        img = np.asarray(img)
        
        return img, label, name

    def __len__(self):
        return self.num_data