import os, sys, pdb
import glob, random
import numpy as np
import itertools
from collections import Counter
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from data_generator.image import ImageDataGenerator
from custom_data_loader import ParallDataWraper, DataLoader

"""
data_weigthed_loader consider the label for specific treatmentss
"""
def data_loader(path, batch_size, imSize):


    train_data_gen_args = dict(
                    horizontal_flip=True,
                    zoom_range=0.2,
                    fill_mode='reflect')

    train_image_datagen = ImageDataGenerator(**train_data_gen_args).flow_from_directory(
                                path+'train/img',
                                class_mode="sparse",
                                target_size=(imSize, imSize),
                                batch_size=batch_size)

    test_image_datagen = ImageDataGenerator().flow_from_directory(
                                path+'test/img',
                                class_mode="sparse",
                                target_size=(imSize, imSize),
                                batch_size=batch_size,
                                seed=1234)

    sys.stdout.flush()
    return train_image_datagen,  test_image_datagen, train_image_datagen.samples, test_image_datagen.samples

def data_loader2(path, batch_size, imSize):


    data_loader_train = DataLoader(path, batch_size, which_set='train', image_size=imSize, shuffle=False, use_extra_data=True)
    data_loader_val = DataLoader(path, batch_size,  which_set='test', image_size=imSize, shuffle=False, use_extra_data=True)
    p_data_loader_train = ParallDataWraper(data_loader_train, batch_size=batch_size, thread=2)
    p_data_loader_val = ParallDataWraper(data_loader_val, batch_size=batch_size, thread=2)

    sys.stdout.flush()
    return p_data_loader_train,  p_data_loader_val, data_loader_train.num_data, data_loader_val.num_data
