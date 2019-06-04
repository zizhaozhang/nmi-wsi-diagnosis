import json
import os, pdb
import string
import numpy as np
import h5py
import random
import scipy.misc as misc
# from .dataset_info import *
# from sklearn.model_selection import StratifiedKFold
from collections import OrderedDict
'''
Used for final
'''
class SETLOADERINBAL():
    def __init__(self, raw_feats, batch_size, shuffle,
                    slide_label, feat_ids):

        self.slide_label = slide_label
        self.slide_list = []
        self.slide_data = OrderedDict()
        self.label_idx_count = {0:[], 1:[]}
        for i, slide in enumerate(raw_feats):
            self.slide_data[slide] = {}
            self.slide_data[slide]['logits'] = raw_feats[slide]['logits']
            self.slide_data[slide]['label'] = int(raw_feats[slide]['label'][0])
            feats = []
            for fid in feat_ids:
                feats.append(raw_feats[slide]['feat'+str(fid)])
            self.slide_data[slide]['feat'] = np.concatenate(feats, axis=1)
            label = self.slide_data[slide]['label']
            assert slide_label[slide] - 1 == label, print('{} is not key'.format(slide))
            self.label_idx_count[label].append(i)
        # it has too be corresponded to self.label_idx_count.
        # it is guaranteed by OrderedDict()
        self.slide_list = [a for a in self.slide_data.keys()]
        self.ifshuffle = shuffle
        self.batch_size = batch_size

        self.num_feat = 200
        self.num_class = 3 #the logit dimension from CNN features


        self.num_data = len(self.slide_list)
        self.order = [a for a in range(self.num_data)]
        print ('\t init data loader')
        if shuffle == True: \
            # training data: augment it to be balanced
            expand_label = min(self.label_idx_count.keys()) # expand the low_grade
            n = [len(a) for a in self.label_idx_count.values()]
            multi = int(max(n) / min(n)) - 1
            self.order += self.label_idx_count[expand_label] * multi
            self.num_data = len(self.order)
            print ('\t --> Balance label. Rebuild order by {} times, totally {} data'.format(multi, self.num_data))
        self.index = 0
        self.shuffle(overflow=False)

    def shuffle(self, overflow=True):

        if self.ifshuffle and (not overflow or self.index + self.batch_size >= self.num_data):
            random.shuffle(self.order)
            # print ('--> shuffing data ...')
        if self.index + self.batch_size >= self.num_data:
            self.index = 0

    def next(self, ids):

        slide_name = self.slide_list[self.order[ids]]
        data = self.slide_data[slide_name]

        self.index += 1
        self.shuffle()

        return data, slide_name


class FEATLOADER:
    def __init__(self, batch_size, sampling_rate,
                    raw_feat_path, groundtruth_root, shuffle=True,
                        feat_dim=4096, feat_ids=[0,1], use_selected_slide=False):

        if use_selected_slide:
            selected_slides = json.load(open('./data/wsi/selected_diagnosis_for_comparsion_100.json'))
            print ('use {} selected diagnosis slides'.format(len(selected_slides)))
        # import pdb; pdb.set_trace()
        # read
        self.raw_feat = {}
        with h5py.File(raw_feat_path+'.h5', "r") as f:
            self.slide_list = f.keys()
            for slide in self.slide_list:
                if use_selected_slide and slide not in selected_slides:
                    continue

                self.raw_feat[slide] = {}
                for k in f[slide].keys():
                    self.raw_feat[slide][k] = f[slide][k][:]

            slide_label = {k:v for k, v in json.load(open(os.path.join(groundtruth_root))).items() if k in self.raw_feat.keys()}

        if use_selected_slide:
            print ('use selected slides')
            print (set(selected_slides) - set([a for a in self.raw_feat.keys()]) )
            assert(len(slide_label) == len(selected_slides))

        self.batch_size = batch_size
        self.sampling_rate = sampling_rate
        self.num_feat = 200 # TODO
        self.sampling_num = int(self.num_feat * self.sampling_rate)
        self.feat_dim = feat_dim # TODO
        self.feat_ids = feat_ids

        np.random.seed(12)
        self.set_holder = dict()

        self.label_map = {
            0: 0,
            1: 1
        }
        self.num_class = len(self.label_map)

        self.set_holder = SETLOADERINBAL(self.raw_feat, self.batch_size, shuffle=shuffle,
                                        slide_label=slide_label,
                                        feat_ids=self.feat_ids)

    def get_iter_epoch(self):
        return int(self.set_holder.num_data / self.set_holder.batch_size)

    def load_batch(self):
        loader = self.set_holder

        X_batch = np.zeros((loader.batch_size, self.feat_dim), np.float32)
        Y_batch = np.zeros(loader.batch_size, np.int32)

        name_list = []
        for c, i in enumerate(range(loader.index, loader.index+loader.batch_size)):
            slide_m, slide_name = loader.next(i)
            # print ('load name '+str(i%len(loader.slide_list)) + ' ', slide_name)
            # print ('sampling ' + slide_name)
            X_batch[c,:], Y_batch[c] = self.sampling_feat(slide_m)
        return X_batch, Y_batch, slide_name

    def load_batch_test(self, duplication):

        loader = self.set_holder

        X_batch = np.zeros((loader.batch_size*duplication, self.feat_dim), np.float32)
        Y_batch = np.zeros(loader.batch_size*duplication, np.int32)
        assert(duplication % self.num_class == 0)
        name_list = []
        for c, i in enumerate(range(loader.index, loader.index+loader.batch_size)):

            slide_m, slide_name = loader.next(i)
            # print ('load name '+str(i%len(loader.slide_list)) + ' ', slide_name)
            # print ('sampling ' + slide_name)
            # generate multiple copies for each data where each copy uses logit of one class as sampling rate
            for d in range(duplication):
                which_lgit = list(self.label_map.values())[d%self.num_class]
                X_batch[c*duplication+d,:], Y_batch[c*duplication+d] = self.sampling_feat(slide_m, use_label=which_lgit)

        return X_batch, Y_batch, slide_name
    def get_run_num(self):
        return self.set_holder.num_data // self.batch_size

    def sampling_feat(self, slide_data, use_label=None):
        logits = slide_data['logits'].copy()
        label =  self.label_map[slide_data['label']]

        if use_label is None:
            logits = logits[:,label]
        else:
            logits = logits[:,use_label]
            # print ('use label '+ str(use_label))

        feats = slide_data['feat']
        p = logits / logits.sum()
        p[p<0.001] = 0 # prevent the errors of multinomial
        indices = np.random.multinomial(self.sampling_num, p)
        # indices = range(200)
        slt_feat = feats[indices, :].copy()
        slt_feat = np.mean(slt_feat, axis=0) # average all feature

        return slt_feat, label
