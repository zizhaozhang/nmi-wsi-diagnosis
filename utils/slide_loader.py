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

# used for extra_cnn_feat.py and mlp_test.py
class SlideLoader():
    def __init__(self, folder, feat_dim, sampling_rate=0.2, imSize=[256,256], slide_label_file='', patch_per_slide=200):

        print ('Configure SlideLoader')
        self.label_map = {
            1: 0,
            2: 1
        }
        self.slide_list = list()
        self.label = []
        self.name_list = []
        self.slide_label = json.load(open(slide_label_file))
        # list all folder
        self.fail_cases = []
        for c, (dirpath, dirnames, filenames) in enumerate(os.walk(folder)):
            if c == 0:
                continue
            slide_name = os.path.split(dirpath)[1]
            if slide_name not in self.slide_label:
                continue
            label = self.label_map[int(self.slide_label[slide_name])]


            slide_list = []
            files = [f for f in filenames if f.endswith(".png")]
            if len(files) < patch_per_slide:
                print('{} has {}(!= 200) data. Skip for now'.format(slide_name, len(files)))
                self.fail_cases.append(slide_name)
                continue
            for filename in files:
                slide_list.append(os.path.join(dirpath, filename))

            self.slide_list.append(slide_list)
            self.label += [label]
            self.name_list += [slide_name]

        self.img_size = imSize
        self.num_data = len(self.label)

        self.feat_dim = feat_dim
        self.num_class = 3

        self.img_per_slide = len(self.slide_list[0])


        self.num_feat = self.img_per_slide
        self.sampling_num = int(self.num_feat * sampling_rate)
        # assert(self.img_per_slide == self.num_feat)
        print ('\t --> search data from ' + folder)
        print ('\t --> {} slides are loaded ...'.format(self.num_data))
        print ('\t --> {} images per slide ...'.format(self.img_per_slide))

        self.index = 0

    # def parse_label(self, slide_name):

    #     if slide_name in self.slide_label.keys():
    #         label = int(self.slide_label[slide_name])
    #     else:
    #         if 'type' in slide_name: # UF data
    #             label = int(slide_name[slide_name.find('type')+4])
    #         else: # TCGA data
    #             label = int(slide_name.split('_')[0])
    #             print ('WARNING: Can not find label of {} from record. Infer the label to be {}'.format(slide_name, label))
    #     return label

    def load_slide_image(self):

        batch = np.zeros((self.img_per_slide, self.img_size[0], self.img_size[1], 3), np.float32)
        label = self.label[self.index]
        slide_name = self.name_list[self.index]
        slide_img_list = []
        slide_img_loc = []
        for c, path in enumerate(self.slide_list[self.index]):
            batch[c] = self.load_img(path)
            slide_img_name = os.path.splitext(os.path.split(path)[1])[0]
            x, y = slide_img_name.split('_')[-2:]
            slide_img_loc.append([int(x), int(y)])
            slide_img_list.append(slide_img_name)

        slide_img_loc = np.array(slide_img_loc)
        self.index += 1
        if self.index == self.num_data:
            self.index = 0


        return batch, label, slide_name, slide_img_loc

    # def sampling_batch(self, batch_feat, batch_logits, duplications, label=None):

    #     def sampling_one(p):
    #         p = p / p.sum()
    #         p[p<0.001] = 0 # prevent the errors of multinomial
    #         indices = np.random.multinomial(self.sampling_num, p)
    #         slt_feat = batch_feat[indices,:].copy()
    #         slt_feat = np.mean(slt_feat, axis=0) # average all feature
    #         return slt_feat

    #     assert(duplications % self.num_class == 0)
    #     X_batch = np.zeros((duplications, self.feat_dim), np.float32)

    #     for i in range(duplications):
    #         logit_p = i%self.num_class
    #         logits = batch_logits[:, logit_p]
    #         X_batch[i] = sampling_one(logits)

    #     return X_batch

    def load_img(self, path):
        img = misc.imread(path)
        # do post processing
        img = misc.imresize(img, size=self.img_size).astype(np.float32)
        # img = img / 255
        # img -= np.reshape(self.rgb_mean, newshape=(1, 1, 3))
        return img
