from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
K = tf.keras.backend
import os
import numpy as np

from sklearn.metrics import confusion_matrix
from scipy.stats import mode
import sys, argparse
from utils.slide_loader import SlideLoader
import deepdish
from classification import classifier
import h5py

parser = argparse.ArgumentParser(description="Settings for extract CNN features")
# setting paramters
# parser.add_argument('--set', type=str, default='test')
parser.add_argument('--folder', type=str, default='segmentor')
parser.add_argument('--data_root', type=str, default='')
parser.add_argument('--save_name', type=str, default='')
parser.add_argument('--slide_label', type=str, default='')
parser.add_argument('--model_name', type=str, default='')

args = parser.parse_args()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)  # set sess before initialize variable...
#####################################################
folder = os.path.join(args.data_root, args.folder)
# name = os.path.join(args.data_root, args.folder+'_inception_feat')
####################################################
## configure data loader
loader = SlideLoader(folder, sampling_rate=0.2, feat_dim=4096, slide_label_file=args.slide_label)

## define image model
img_shape = [256, 256]
feed_img_batch = tf.placeholder(tf.float32, shape=[None]+img_shape+[3])
print ('--> init the image model ...')
feat_model, _ = classifier.get_main_network('inception', input_tensor=feed_img_batch, num_classes=3, use_weights=False)
mixed_layer = [a for a in feat_model.layers if 'mixed' in a.name ]
for l in mixed_layer:
    print (l.name, l.output.shape)
"""
    extract features
    pay attention to which_layers
"""
logit = tf.keras.layers.Lambda(K.softmax)(feat_model.output)
which_layers = [-1,-3,-5,-6]
feats = []
for i in which_layers:
    print('extract {}: {}'.format(mixed_layer[i].name, mixed_layer[i].output.shape))
    f1_conv = mixed_layer[i].output # 2048
    f1 = tf.keras.layers.GlobalAveragePooling2D()(f1_conv)
    feats.append(f1)
'''
extract mixed10: (?, 6, 6, 2048)
extract mixed9: (?, 6, 6, 2048)
extract mixed8: (?, 6, 6, 1280)
extract mixed7: (?, 14, 14, 768)
'''

''' Main '''
with sess.as_default():
    #classification_pretrained_data1_checkpoint_2_best_87.9

    feat_model.load_weights('./checkpoints/classification/trained_model/{}.h5'.format(args.model_name), by_name=True)

    with h5py.File('{}/{}_{}_cnnfeat_wloc.h5'.format(args.data_root, args.save_name, args.model_name), "w") as f:
        print ('{}/{}_{}_cnnfeat_wloc.h5'.format(args.data_root, args.save_name, args.model_name))
        for i in range(loader.num_data):
            batch, label, slide_name, slide_img_loc = loader.load_slide_image()
            print('processing {}/{} slide {} with label {} '.format(i, loader.num_data, slide_name, label))
            feed_dict = {
                            feed_img_batch: batch,
                            K.learning_phase(): False # not very sure about it
                        }
            batch_logits, batch_feats1, batch_feats2, batch_feats3, batch_feats4 = sess.run([logit] + feats, feed_dict=feed_dict)
            batch_feats = [batch_feats1, batch_feats2, batch_feats3, batch_feats4]

            grp = f.create_group(slide_name)
            for gname, batch_f in enumerate(batch_feats):
                grp.create_dataset("feat"+str(gname), batch_f.shape)
                grp["feat"+str(gname)][:] = batch_f
            grp.create_dataset("logits", batch_logits.shape)
            grp["logits"][:] = batch_logits
            grp.create_dataset("label", (1,))
            grp["label"][:] = label
            grp.create_dataset("loc", slide_img_loc.shape)
            grp["loc"][:] = slide_img_loc

    #deepdish.io.save('{}/{}.h5'.format(args.data_root, args.save_name),{'feats': FEATS, 'slide_dict': SLIDE_DICT, 'fail_slides': loader.fail_cases})
