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

    batch_captions = np.transpose(batch_captions, (1,0,2))
    batch_captions = np.reshape(batch_captions,newshape=(batch_captions.shape[0]*batch_captions.shape[1], batch_captions.shape[2]))

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

    def __init__(self, opt, which_set, image_size=[256,256,3], shuffle=True):

        self.batch_size = opt.batch_size
        self.dataset_dir = opt.dataset_dir
        self.imdir = os.path.join(self.dataset_dir, 'Images')
        self.num_feature = 6
        self.max_subseq_len = 15 + 1 # it is manually computed. <end> (last) is counted
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
        
        img = misc.imread(os.path.join(self.imdir, name+'.png'))
        img = misc.imresize(img, size=self.img_size).astype(np.float32)
        caption = self.anno[name]['caption'][capid]
        label = self.anno[name]['label']
        if label == 0: label = 3 # make normal being insufficient information
        label = label - 1 # start from [0, num_class)
        # split captions to sentences
        sentences = caption.rstrip('.').replace(',','').split('. ')
        assert(len(sentences) == self.num_feature, 'the number of sentence is not correct in [{}]'.format(caption))
        sent_tokens = [] # convert to tokens for all sentences
        for s, sentence in enumerate(sentences):
            tokens = nltk.tokenize.word_tokenize(str(sentence).lower())
            tokens.append('<end>') # add stop indictor
            tokens = [self.keywords[s]] + tokens # adding feature indictor
            tmp = [self.vocab(token) for token in tokens]
            sent_tokens.append(tmp)
        # +1 to add feature indictor at the head
        caption = np.ones((self.num_feature, self.max_subseq_len+1), np.int32) * self.PAD_TOKEN
        for j, tmp in enumerate(sent_tokens):
                caption[j,:len(tmp)]= tmp
        paragraph = ' '.join(sentences).lower()

        return img, caption, label, paragraph, name

    def load_batch(self):
        # use if do not use Parallel mode
        batch_size = self.batch_size
        
        # load 
        batch_imgs = []
        batch_labels = []
        batch_tokens = []
        batch_captions = []
        batch_sentences = []
        batch_names = []
        batch_gt_sents = []
        loc_max_subseq_len = 0 # find the max len in this batch
        for c, i in enumerate(range(self.index, self.index+batch_size)):
            id = self.order[i]
            capid = int(id / len(self.anno))
            imid = int(id / self.num_anno_per_img)

            name = self.name_list[imid]
            
            img = misc.imread(os.path.join(self.imdir, name+'.png'))
            img = misc.imresize(img, size=self.img_size).astype(np.float32)
            caption = self.anno[name]['caption'][capid]
            label = self.anno[name]['label']
            if label == 0: label = 3 # make normal being insufficient information
            label = label - 1 # start from [0, num_class)
            # split captions to sentences
            sentences = caption.rstrip('.').replace(',','').split('. ')
            assert(len(sentences) == self.num_feature, 'the number of sentence is not correct in [{}]'.format(caption))
            sent_tokens = [] # convert to tokens for all sentences
            for s, sentence in enumerate(sentences):
                tokens = nltk.tokenize.word_tokenize(str(sentence).lower())
                tokens.append('<end>') # add stop indictor
                tokens = [self.keywords[s]] + tokens # adding feature indictor
                tmp = [self.vocab(token) for token in tokens]
                if len(tmp) > loc_max_subseq_len: loc_max_subseq_len = len(tmp)
                sent_tokens.append(tmp)

            batch_sentences.append(sentences)
            batch_tokens.append(sent_tokens)
            batch_imgs.append(img)
            batch_labels.append(label)
            batch_names.append(name)
            batch_gt_sents.append(' '.join(sentences).lower())

        self.index += self.batch_size
        if self.index + self.batch_size > self.num_data:
            self.shuffle_order() # shuffle if needed

        batch_labels = np.array(batch_labels)
        batch_img = np.array(batch_imgs)

        # organize batch_token to batch_captions in matrix
        batch_captions = np.ones((batch_size, self.num_feature, self.max_subseq_len+1), np.int32) * self.PAD_TOKEN # +1 for the feature_indictor
        for i, tokens in enumerate(batch_tokens):
            for j, tmp in enumerate(tokens):
                batch_captions[i,j,:len(tmp)]= tmp

        # permute is necessary to make batch order in the [batch1_fea1, batch2_feat1, ... , batch1_feat2, batch2_feat2 ...] order 
        batch_captions = np.transpose(batch_captions, (1,0,2))
        batch_captions = np.reshape(batch_captions,newshape=(batch_captions.shape[0]*batch_captions.shape[1], batch_captions.shape[2]))

        return batch_imgs, batch_captions, batch_labels, batch_gt_sents, batch_names 

    def get_iter_epoch(self, split='train'):
        return int(self.num_data / self.batch_size)
    
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

# used for extra_cnn_feat.py and mlp_test.py
class SlideLoader():
    def __init__(self, folder, sampling_rate=0.2, slide_label_file=''):

        print ('Configure SlideLoader')
        self.slide_list = list()
        self.label = []
        self.name_list = []
        self.slide_label = json.load(open(slide_label_file))
        # list all folder
        for c, (dirpath, dirnames, filenames) in enumerate(os.walk(folder)):
            if c == 0: continue
            slide_name = os.path.split(dirpath)[1]
            label = self.parse_label(slide_name)

            slide_list = []
            for filename in [f for f in filenames if f.endswith(".png")]:
                slide_list.append(os.path.join(dirpath, filename))
                
            self.slide_list.append(slide_list)  
            self.label += [label]
            self.name_list += [slide_name]

        self.img_size = [299,299]
        self.num_data = len(self.label)

        self.feat_dim = 2048 + 1024
        self.num_class = 3
        
        self.img_per_slide = len(self.slide_list[0])
        for v in self.slide_list:
            # if len(v) != self.img_per_slide:
                # print (len(v), v)
            assert(len(v) == self.img_per_slide)
            
        self.num_feat = self.img_per_slide
        self.sampling_num = int(self.num_feat * sampling_rate)
        # assert(self.img_per_slide == self.num_feat)
        print ('\t --> search data from ' + folder)
        print ('\t --> {} slides are loaded ...'.format(self.num_data))
        print ('\t --> {} images per slide ...'.format(self.img_per_slide))

        self.index = 0

    def parse_label(self, slide_name):
        
        if self.slide_label.has_key(slide_name):
            label = int(self.slide_label[slide_name])
        else:
            if 'type' in slide_name: # UF data
                label = int(slide_name[slide_name.find('type')+4])
            else: # TCGA data
                label = int(slide_name.split('_')[0])
                print ('WARNING: Can not find label of {} from record. Infer the label to be {}'.format(slide_name, label))
        return label

    def load_slide_image(self):

        batch = np.zeros((self.img_per_slide, self.img_size[0], self.img_size[1], 3), np.float32)
        label = self.label[self.index]
        slide_name = self.name_list[self.index]
        slide_img_list = []

        for c, path in enumerate(self.slide_list[self.index]):
            batch[c] = self.load_img(path)
            slide_img_list.append(os.path.splitext(os.path.split(path)[1])[0])

        self.index += 1
        if self.index == self.num_data:
            self.index = 0

        return batch, label, slide_name, slide_img_list
    
    def sampling_batch(self, batch_feat, batch_logits, duplications, label=None):

        def sampling_one(p):
            p = p / p.sum()
            p[p<0.001] = 0 # prevent the errors of multinomial
            indices = np.random.multinomial(self.sampling_num, p)
            slt_feat = batch_feat[indices,:].copy()
            slt_feat = np.mean(slt_feat, axis=0) # average all feature
            return slt_feat
        
        assert(duplications % self.num_class == 0)
        X_batch = np.zeros((duplications, self.feat_dim), np.float32)

        for i in range(duplications):
            logit_p = i%self.num_class
            logits = batch_logits[:, logit_p]
            X_batch[i] = sampling_one(logits)
            
        return X_batch

    def load_img(self, path):
        img = misc.imread(path)
        # do post processing
        img = misc.imresize(img, size=self.img_size).astype(np.float32)
        img = img / 255
        # img -= np.reshape(self.rgb_mean, newshape=(1, 1, 3))
        return img

# load a folder of images with labels. Used to test MDNet in test.py
class ImageFolderLoader():
    def __init__(self, img_dir, batch_size):

        self.imlist = sorted(glob.glob(os.path.join(img_dir, 'images/*.png')))  # sorted
        self.num_data = len(self.imlist)
        self.order = range(self.num_data)
        self.img_size = [299,299]
        self.index = 0
        self.batch_size = batch_size
        assert(float(self.num_data) % self.batch_size == 0)
        self.feat_topic_vector = np.array([ 1, 6, 12, 16, 21, 24 ]).reshape([6,1])

    def load_batch(self):

        batch_img = np.zeros((self.batch_size, self.img_size[0], self.img_size[1], 3), np.float32)

        batch_feat_topic = np.tile(self.feat_topic_vector.copy(), [1, self.batch_size])
        batch_feat_topic = batch_feat_topic.reshape(-1)
        num_feat = self.feat_topic_vector.size
        
        batch_label = np.zeros((self.batch_size*num_feat, 11), np.int32) 
        batch_label[:,0] = batch_feat_topic

        name_list = []
        for c, i in enumerate(range(self.index, self.index + self.batch_size)):
            name = os.path.split(self.imlist[i])[1]
            name = os.path.splitext(name)[0]
            name_list.append(name)
            batch_img[c] = self.load_img(self.imlist[i])
            
        self.index += self.batch_size

        if self.index + self.batch_size > self.num_data:
            self.index = 0
        
        return batch_img, batch_label, name_list

    def get_iter_epoch(self):
        return self.num_data / self.batch_size

    def load_img(self, path):
        img = misc.imread(path)
        # do post processing
        img = misc.imresize(img, size=self.img_size).astype(np.float32)
        img = img / 255
        # img -= np.reshape(self.rgb_mean, newshape=(1, 1, 3))
        return img