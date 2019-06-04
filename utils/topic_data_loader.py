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
from .vocabulary import Vocabulary
import torchvision.transforms as transforms
from PIL import Image

def pack_data(outputs):
    batch_imgs = []
    batch_labels = []
    batch_tokens = []
    batch_captions = []
    batch_gt_sents = []
    batch_names = []
    stop_indictors = []

    for (img, (caption, stop_indictor), label, paragraph, name) in outputs:
        batch_imgs.append(img)
        batch_labels.append(label)
        batch_names.append(name)
        batch_captions.append(caption)
        batch_gt_sents.append(paragraph)
        stop_indictors.append(stop_indictor)

    batch_labels = np.array(batch_labels)
    batch_imgs = np.array(batch_imgs)
    batch_captions = np.array(batch_captions) #(batch_size, num_feat, max_subseq_len)
    stop_indictors = np.array(stop_indictors) #(batch_size, num_feat)

    batch_captions = np.transpose(batch_captions, (1,0,2)) #(num_feat, batch_size, max_subseq_len)
    batch_captions = np.reshape(batch_captions, newshape=(batch_captions.shape[0]*batch_captions.shape[1], batch_captions.shape[2]))
    
    return batch_imgs, (batch_captions, stop_indictors), batch_labels, batch_gt_sents, batch_names 

class ParallDataWraper():
    def __init__(self, loader, batch_size, thread=0):
        self.loader = loader
        assert loader.shuffle == False, 'Shuffle in loader should be False'
        self.batch_size = batch_size
        self.pool = Pool(thread)
        self.create_loader(self.loader.num_data)

    def create_loader(self, num):
        ids = [a for a in range(num)]
        random.shuffle(ids)
        self.targets = self.pool.imap(self.loader.next, (id for id in ids))

    def reset(self):
        self.create_loader(self.loader.num_data)

    def get_iter_epoch(self):
        return self.loader.get_iter_epoch()

    def load_batch(self):
        all_outputs = []
        for i in range(self.batch_size):
            try:
                outputs = self.targets.__next__()
            except StopIteration:
                self.create_loader(self.loader.num_data)
                outputs = self.targets.__next__()
            all_outputs.append(outputs)
        
        return pack_data(all_outputs)


class DataLoader():

    def __init__(self, opt, which_set, image_size=256, shuffle=True, use_augmentation=True):
        print('init topic data_loader ...')
        
        self.batch_size = opt.batch_size
        self.dataset_dir = opt.dataset_dir
        self.imdir = os.path.join(self.dataset_dir, 'Images')
        self.num_feature = 6
        self.max_subseq_len = 15 + 1 # it is manually computed. <end> (last) is counted
        self.img_size = image_size
        self.keywords = ['nuclear_feature', 'nuclear_crowding', 'polarity',
                                'mitosis', 'nucleoli', 'conclusion'] 
        self.stop_label = 2
        self.anno = json.load(open(os.path.join(self.dataset_dir, '{}_annotation.json'.format(which_set)), 'r'))
        self.name_list = list(self.anno.keys())
        # setup iterator
        self.index = 0
        if which_set == 'train':
            self.num_anno_per_img = 5 # number of annotation per sample
        else:
            # during test we only need the name no need to repeat all annotations
            self.num_anno_per_img = 1
            # print('WARNING: I change num_anno_per_img to 5 for retrieval experiment')

        if which_set == 'train' and use_augmentation:
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

        self.num_data = len(self.anno) * self.num_anno_per_img
        self.order = [a for a in range(0, self.num_data)]
        self.shuffle = shuffle
        self.vocab = pickle.load(open('utils/vocab_bladderreport.pkl','rb'))
        self.END_TOKEN = self.vocab.word2idx['<end>']
        self.PAD_TOKEN = self.vocab.word2idx['<pad>'] # PAD_TOKEN is used for not supervison
        self.vocab_size = len(self.vocab.word2idx)

        self.shuffle_order()
        
        print('\t {} samples from the {} set'.format(len(self.order), which_set))
    

    def shuffle_order(self):
        if self.shuffle:
            random.shuffle(self.order)
            print ('--> shuffling data ...')
        self.index = 0
    
    def next(self, idx):
        id = self.order[idx]
        capid = int(id / len(self.anno))
        imid = int(id / self.num_anno_per_img)

        name = self.name_list[imid]
        
        raw_img = Image.open(os.path.join(self.imdir, name+'.png'))
        img = self.transform(raw_img)
        img = np.asarray(img)

        caption = self.anno[name]['caption'][capid]
        label = self.anno[name]['label']
        if label == 0: label = 3 # treat normal as insufficient information
        label = label - 1 # start from [0, num_class) in json file 0 is normal originally
        # split captions to sentences
        sentences = caption.rstrip('.').replace(',','').split('. ')
        assert(len(sentences) == self.num_feature, 'the number of sentence is not correct in [{}]'.format(caption))
        sent_tokens = [] # convert to tokens for all num_feature sentences
        paragraph = []
        for s, sentence in enumerate(sentences):
            # if feature (except conclusion) is insufficient information, do not output it
            # but the conclusion (last one) is insufficient information, we still output it
            if 'insufficient' in sentence and s < (len(sentences)-1): 
                continue
            tokens = nltk.tokenize.word_tokenize(str(sentence).lower())
            paragraph.append(str(sentence))
            tokens.append('<end>') # add stop indictor
            tmp = [self.vocab(token) for token in tokens]
            sent_tokens.append(tmp)
        # +1 to add feature indictor at the head
        caption = np.ones((self.num_feature, self.max_subseq_len), np.int32) * self.PAD_TOKEN # 1 for end token
        stop_indictor = np.ones(self.num_feature, np.int32) * self.PAD_TOKEN
        assert(self.PAD_TOKEN == 0)
        # sent_tokens length varies, so the rest is padded with -1
        for j, tmp in enumerate(sent_tokens):
                caption[j,:len(tmp)]= tmp
                stop_indictor[j] = 1 # continue indictor
        stop_indictor[j] = self.stop_label # stop indictor
        paragraph = ' '.join(paragraph).lower()
        
        return img, (caption, stop_indictor), label,  paragraph, name


    def get_iter_epoch(self):
        return int(self.num_data / self.batch_size)
        
    def add_stops(self, txt_stops):
        # the model may also predict if some feature will be described
        # if it is set, we should control the conver_to_text_list based on stop_label
        self.txt_stops = txt_stops

    def convert_to_text_list(self, label):
        # label [batch_size, feature_num, seq_len]

        label = label.astype(np.int32)
        text_list = []
        text_list_verbose = []
        if len(label.shape) == 2:
            label = np.expand_dims(label,axis=0)
        assert(len(label.shape) == 3)
        batch_feature_len = []
        # for all samples
        for k in range(label.shape[0]):
            text = []
            text_vb = []
            tmp_idx = []
            # for all feature
            for i in range(label.shape[1]):
                if hasattr(self, 'txt_stops'):
                    if i!=0 and self.txt_stops[k][max(0,i-1)] == self.stop_label:
                        break
                subtext = []
                # for each time step
                for s in range(label.shape[2]):
                    if label[k][i][s] != self.END_TOKEN:
                        subtext += [self.vocab.idx2word[label[k][i][s]]]
                    else:
                        break
                    # if label[k][i][s] == self.word_to_idx['.']:
                    #     break
                text += subtext
                # text_vb = text_vb + [self.keywords[i]] + subtext
                tmp_idx.append(len(subtext))

            batch_feature_len.append(tmp_idx)
            text_list.append(' '.join(text[:]))
            # text_list_verbose.append(' '.join(text_vb[:]))

        return text_list, batch_feature_len        


# load a folder of images with labels. Used to test MDNet in test.py
class ImageFolderLoader():
    def __init__(self, img_dir, batch_size, image_size=256, extension='.png'):
        
        
        self.num_feature = 6
        self.max_subseq_len = 15 + 1 # it is manually computed. <end> (last) is counted

        self.imlist = []
        folders = []
        for root, _, files in os.walk(img_dir):
            if root == img_dir: continue
            if '.' in os.path.split(root)[1]: continue
            images = [os.path.join(root, file) for file in files if extension in file]
            if len(images) == 0: continue
            self.imlist.extend(images)
            folders.append(root)
        self.img2label = {**json.load(open('wsi_cls/test_label_293.json','r')), **json.load(open('wsi_cls/train_label_620.json','r'))}
        self.num_data = len(self.imlist)
        self.order = range(self.num_data)
        self.img_size = image_size
        self.index = 0
        self.batch_size = batch_size
        assert float(self.num_data) % self.batch_size == 0, 'use a dividable batch_size wrt the data amount '+str(self.num_data)
        # self.feat_topic_vector = np.array([ 1, 6, 12, 16, 21, 24 ]).reshape([6,1])
        self.transform = transforms.Compose([ 
                        transforms.Resize(299),
                        transforms.CenterCrop(image_size),
                                    ])
        self.vocab = pickle.load(open('utils/vocab_bladderreport.pkl','rb'))
        self.vocab_size = len(self.vocab.word2idx)
        self.END_TOKEN = self.vocab.word2idx['<end>']
        self.PAD_TOKEN = self.vocab.word2idx['<pad>'] # PAD_TOKEN is used for not supervison
        self.stop_label = 2
        print('ImageFolderLoader \n \t Find {} folder and {} images'.format(len(folders), len(self.imlist)))
        print('\t {} batches'.format(self.get_iter_epoch()))
        
    def load_batch(self):

        batch_img = np.zeros((self.batch_size, self.img_size, self.img_size, 3), np.float32)

        # batch_feat_topic = np.tile(self.feat_topic_vector.copy(), [1, self.batch_size])
        # batch_feat_topic = batch_feat_topic.reshape(-1)
        # num_feat = self.feat_topic_vector.size
        
        batch_imglabel = np.zeros((self.batch_size, ), np.int32) 

        name_list = []
        for c, i in enumerate(range(self.index, self.index + self.batch_size)):
            path, name = os.path.split(self.imlist[i])
            slide_name = os.path.split(path)[1]
            if 'images' in  slide_name:
                slide_name = os.path.split(os.path.split(path)[0])[1]
            name = os.path.splitext(name)[0]
            name_list.append(name)
            
            img = Image.open(self.imlist[i])
            batch_img[c] = np.asarray(self.transform(img))
            batch_imglabel[c] = self.img2label[slide_name] - 1

        self.index += self.batch_size

        if self.index + self.batch_size > self.num_data:
            self.index = 0
        
        return batch_img, batch_imglabel,  name_list

    def get_iter_epoch(self):
        return self.num_data // self.batch_size

    def add_stops(self, txt_stops):
        # the model may also predict if some feature will be described
        # if it is set, we should control the conver_to_text_list based on stop_label
        self.txt_stops = txt_stops

    def convert_to_text_list(self, label):
        # label [batch_size, feature_num, seq_len]

        label = label.astype(np.int32)
        text_list = []
        text_list_verbose = []
        if len(label.shape) == 2:
            label = np.expand_dims(label,axis=0)
        assert(len(label.shape) == 3)
        batch_feature_len = []
        # for all samples
        for k in range(label.shape[0]):
            text = []
            text_vb = []
            tmp_idx = []
            # for all feature
            for i in range(label.shape[1]):
                if hasattr(self, 'txt_stops'):
                    if i!=0 and self.txt_stops[k][max(0,i-1)] == self.stop_label:
                        break
                subtext = []
                # for each time step
                for s in range(label.shape[2]):
                    if label[k][i][s] != self.END_TOKEN:
                        subtext += [self.vocab.idx2word[label[k][i][s]]]
                    else:
                        break
                    # if label[k][i][s] == self.word_to_idx['.']:
                    #     break
                text += subtext
                # text_vb = text_vb + [self.keywords[i]] + subtext
                tmp_idx.append(len(subtext))

            batch_feature_len.append(tmp_idx)
            text_list.append(' '.join(text[:]))
            # text_list_verbose.append(' '.join(text_vb[:]))

        return text_list, batch_feature_len  