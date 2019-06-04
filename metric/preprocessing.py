# convert description to the format for metric evaluation
# need to trim some common words in remove_common_words 
# run this function to convert bladder_test_caption
# Zizhao @ UF 

import numpy as np
import json
from json import encoder
import os
import re
import glob 

def remove_chars(sent):
        sent = sent.replace(' .', '')
        sent = sent.replace('.', '')
        sent = sent.replace(' :', '')
        sent = sent.replace(':', '')
        sent = sent.lstrip(' ')
        return sent
        
def remove_common_words(sents):
    # keywords = ['nuclear', 'features', 'crowding', 'polarity', 
    #             'nucleoli', 'is', 'nuclear_feature', 'nuclear_crowding','polarity','mitosis','nucleoli','conclusion', 
    #             # 'are', 'no', 'not', 'is'
    #             ]
    keywords = ['nuclear_feature', 'nuclear_crowding','polarity','mitosis','nucleoli','conclusion']
    
    querywords = sents.split()
    resultwords = list()
    for word in querywords:
        if word not in keywords:
            resultwords.append(word)
        else:
            keywords = keywords[1:] # only delete once
    sents = ' '.join(resultwords)
        
    return sents

# an extreme test
# def remove_common_words(sents):
#     keywords = ['normal','mild', 'moderate', 'severe',
#                 'no',
#                 'not','partially','completely',
#                 'infrequent', 'frequent','rare', 
#                 'prominent','absent','inconspicuous',
#                 'insufficient', 'information',
#                 'normal', 'lg/punlmp', 'hg', 
#                 ]
#     # keywords = ['nuclear_feature', 'nuclear_crowding','polarity','mitosis','nucleoli','conclusion']
    
#     querywords = sents.split()
#     resultwords = list()
#     for word in querywords:
#         if word in keywords:
#             resultwords.append(word)
#         # else:
#         #     keywords = keywords[1:] # only delete once
#     sents = ' '.join(resultwords)
        
#     return sents

# currently I choose to remove all non-alphabetical characters (. and :)
def json_to_sentence(struct, strip_common):

    keywords = ['nuclear_feature','nuclear_crowding','polarity',
                            'mitosis','nucleoli','conclusion']
    sent = ''
    for keyword in keywords:
        if sent != '': sent = sent+ ' ' 
        sent += keyword # first sent -- do we need the keyword in the sentence?
        item = struct[keyword][1] # 0 is the sentence label
        item = item.lower()
        item = item.strip('.')
        # a simple post processing 
        if item == "lg / punlmp":
            item = item.replace(" ", "")
        sent = sent + ' ' + item
    sent = sent.rstrip(' ')
    if strip_common:
        sent = remove_common_words(sent)
    return sent

# convert indivisual description json file to a single one
def merge_json(data_root, strip_common=True):
    # if 'test' in list_name:
    #     mode = 'test'
    # if 'train' in list_name:
    #     mode = 'train'
    # if 'val' in list_name:
    #     mode = 'val'
    # load data Index
    # with open(data_root+list_name,'r') as f:
    #     files = [p.split(',')[0] for p in f.readlines()]

    # load all
    gt_path = os.path.join(data_root, 'Annotation/')
    mode = 'dataset'
    raw_files = glob.glob(gt_path+'*.json')
    files = [os.path.split(p)[1][:-5] for p in raw_files]


    arxiv = list()
    for i, item in enumerate(files):
        item = item.strip()
        tmp = dict()
        tmp['image_id'] = item
        tmp['id'] = i+1 #begin from 1
        with open(gt_path+item+'.json') as f:
            structs = json.load(f)
        sentences = list()
        for struct in structs:
            sentence = json_to_sentence(struct,strip_common)
            sentences.append(sentence)
        tmp['caption'] =  sentences
        arxiv.append(tmp)
    if strip_common:
        resFileName = 'bladder_'+mode+'_caption_striped.json'
    else:
        resFileName = 'bladder_'+mode+'_caption.json'
    json.dump(arxiv, open(resFileName, 'w'))
    print ('write '+resFileName)


if __name__ == "__main__":
    # merge the groundtruth json file into a single on and save under evaluation/ folder for evaluation
    data_root = '../../datasets/bladder/data_organized_multiref/'
    merge_json(data_root)
