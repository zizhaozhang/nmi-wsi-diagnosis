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
import torchvision.transforms as transforms
from PIL import Image
from .vocabulary import Vocabulary

def pack_data(outputs):
    batch_imgs = []
    batch_labels = []
    batch_tokens = []
    batch_captions = []
    batch_gt_sents = []
    batch_names = []
    for (img, caption, label, paragraph, name) in outputs:
        batch_imgs.append(img)
        batch_labels.append(label)
        batch_names.append(name)
        batch_captions.append(caption)
        batch_gt_sents.append(paragraph)


    batch_labels = np.array(batch_labels)
    batch_imgs = np.array(batch_imgs)
    batch_captions = np.array(batch_captions) #(batch_size, num_feat, max_subseq_len)
    # reshape to the longest path
    # batch_captions = np.reshape(batch_captions,newshape=(batch_captions.shape[0],batch_captions.shape[1]*batch_captions.shape[2]))

    return batch_imgs, batch_captions, batch_labels, batch_gt_sents, batch_names 

class ParallDataWraper():
    def __init__(self, loader, batch_size, thread=0):
        self.loader = loader
        self.batch_size = batch_size
        self.pool = Pool(thread)
        self.create_loader(self.loader.num_data)

    def create_loader(self, num):
        # print ('--> remap and shuffle iterator')
        ids = [a for a in range(num)]
        random.shuffle(ids)
        self.targets = self.pool.imap(self.loader.next, (id for id in ids))

    def get_iter_epoch(self):
        return self.loader.get_iter_epoch()

    def reset(self):

        self.create_loader(self.loader.num_data)
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

        self.batch_size = opt.batch_size
        self.dataset_dir = opt.dataset_dir
        self.imdir = os.path.join(self.dataset_dir, 'Images')
        self.num_feature = 6
        self.max_subseq_len = 58 + 1 # it is manually computed. <end> (last) is counted
        self.img_size = image_size
        self.keywords = ['nuclear_feature', 'nuclear_crowding', 'polarity',
                                'mitosis', 'nucleoli', 'conclusion'] 

        self.anno = json.load(open(os.path.join(self.dataset_dir, '{}_annotation.json'.format(which_set)), 'r'))
        self.name_list = list(self.anno.keys())
        # setup iterator
        self.index = 0
        if which_set == 'train':
            self.num_anno_per_img = 5 # number of annotation per sample
        else:
            # during test we only need the name no need to repeat all annotations
            self.num_anno_per_img = 1
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
        self.shuffle = which_set == 'train'
        self.vocab = pickle.load(open('utils/vocab_bladderreport.pkl','rb'))
        self.END_TOKEN = self.vocab.word2idx['<end>']
        self.PAD_TOKEN = self.vocab.word2idx['<pad>']
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
        if label == 0: label = 3 # make normal being insufficient information
        label = label - 1 # start from [0, num_class)

        sentence = caption.replace('.','')
        tokens = nltk.tokenize.word_tokenize(str(sentence).lower())
        tokens.append('<end>') # add stop indictor

        ## split captions to sentences
        sentences = caption.rstrip('.').replace(',','').split('. ')
        assert(len(sentences) == self.num_feature, 'the number of sentence is not correct in [{}]'.format(caption))
        ## sent_tokens = [] # convert to tokens for all sentences
        tmp = []
        for s, sentence in enumerate(sentences):
            if s != self.num_feature - 1 and 'insufficient' in sentence:
                continue
            tmp.append(sentence)
        
        sentence = ' '.join(tmp)
        tokens = nltk.tokenize.word_tokenize(str(sentence).lower())
        tokens.append('<end>') # add stop indictor
        caption = np.ones(self.max_subseq_len, np.int32) * self.PAD_TOKEN
        caption[:len(tokens)]= [self.vocab(token) for token in tokens]
        paragraph = sentence.lower()


        return img, caption, label, paragraph, name

    def get_iter_epoch(self, split='train'):
        return int(self.num_data / self.batch_size)
    
    def convert_to_text_list(self, label):
        # label [batch_size, seq_len]

        label = label.astype(np.int32)
        text_list = []

        batch_feature_len = []
        # for all samples
        for k in range(label.shape[0]):
            text = []
            tmp_idx = []
            # for all feature
            subtext = []
            # for each time step
            for s in range(label.shape[1]):
                if label[k][s] != self.END_TOKEN:
                    subtext += [self.vocab.idx2word[label[k][s]]]
                else:
                    break

            batch_feature_len.append(subtext)
            text_list.append(' '.join(subtext[:]))

        return text_list, batch_feature_len        

