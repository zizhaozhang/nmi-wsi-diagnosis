import os, sys, pdb
import tensorflow as tf
# # Adding local Keras
# HOME_DIR = os.path.expanduser('~')
# keras_version = 'keras_pingpong'
# KERAS_PATH = os.path.join(HOME_DIR, 'Github', keras_version)
# sys.path.insert(0, KERAS_PATH)
# sys.path.insert(0, os.path.join(KERAS_PATH, 'keras'))
# sys.path.insert(0, os.path.join(KERAS_PATH, 'keras', 'layers'))
import keras
from keras.models import Sequential, load_model, Model

import numpy as np
import matplotlib.pyplot as plt
from skimage import io, transform
import argparse, glob
import deepdish as dd
from collections import OrderedDict


def set_args():
    parser = argparse.ArgumentParser(description="Settings for finetuning classifier")
    # setting paramters
    parser.add_argument('-g', '--gpu_serial_num', type=str, default='0')
    parser.add_argument('-m', '--model_name',     type=str, default='inceptionv3')
    parser.add_argument('-r', '--input_row',      type=int, default=299)
    parser.add_argument('-c', '--input_col',      type=int, default=299)
    parser.add_argument('--training_dir',         type=str, default='./FeaSamples/')
    parser.add_argument('--feas_name',            type=str, default='bladder8000feas.h5')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = set_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_serial_num
    print("Tensorflow version: {}".format(tf.__version__))
    print("Keras version: {}".format(keras.__version__))

    # setting model
    weights_path = os.path.join(os.getcwd(), 'PatientsBestModel', args.model_name+'_weights_best.h5')
    ft_model = load_model(weights_path)

    # Get weights
    fc07_weights = ft_model.get_layer('fc07').get_weights()
    # Get intermediate results
    model = Model(input=ft_model.input, output=[ft_model.get_layer('mixed10').output,
                                                ft_model.get_layer('fc06').output,
                                                ft_model.get_layer('fc07').output])

    # # Geting all image names
    # phase_names = ['train', 'val']
    # disease_names = ['0', '1', '2']
    # filelist = []
    # # get all filenames
    # for phase in phase_names:
    #     for disease in disease_names:
    #         cur_path = os.path.join(args.training_dir, phase, 'img', disease, '*.png')
    #         filelist.extend(glob.glob(cur_path))

    filelist = glob.glob(os.path.join(args.training_dir, '*.png'))

    fea_dict = OrderedDict()
    total_num = len(filelist)
    for idx, img_path in enumerate(filelist):
        img = io.imread(img_path)
        img = transform.resize(img, (args.input_row, args.input_col))
        img4d = np.expand_dims(img, axis=0)
        mixed10, fc06, fc07 = model.predict(img4d)

        filename = os.path.splitext(os.path.basename(img_path))[0]

        fea2048 = mixed10.mean(axis=(0, 1, 2))
        fea1024 = fc06.mean(axis=(0, 1, 2))
        fea4 = fc07.mean(axis=0)
        type_pos = filename.find('type')
        fea1 = int(filename[type_pos+4:type_pos+5])
        fea_dict[filename] = [fea2048, fea1024, fea4, fea1]

        print("{}/{} processed.".format(idx+1, total_num))

    # save all features of 8000 images to h5 file
    dd.io.save(args.feas_name, fea_dict)
