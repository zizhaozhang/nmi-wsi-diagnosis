'''
 * @author [Zizhao Zhang]
 * @email [zizhao@cise.ufl.edu]
 * @create date 2017-05-19 03:06:32
 * @modify date 2017-05-19 03:06:32
 * @desc [description]
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

SEED=0 # set set to allow reproducing runs
import numpy as np
np.random.seed(SEED)
import tensorflow as tf
tf.set_random_seed(SEED)

import os, sys, time, math
import scipy.misc as misc
from classifier import get_main_network
from data_gen import data_loader

from util import VIS
# configure opts
from opts import *

# configuration session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
# from keras import backend as K
K = tf.keras.backend
#K.clear_session()
K.set_session(sess) #
#K.set_learning_phase(False)

# save and compute metrics
vis = VIS(save_path=opt.checkpoint_path)

''' Users defined data loader (with train and test) '''
img_shape = [opt.imSize, opt.imSize]
train_generator, test_generator, train_samples, test_samples = data_loader(opt.data_path, opt.batch_size, imSize=opt.imSize)

test_iter = int(test_samples / opt.batch_size)

# define input holders
label = tf.placeholder(tf.int32, shape=[None])
img = tf.placeholder(tf.float32, shape=[None]+img_shape+[3])

# define model
with tf.name_scope('inception'):    
    model, pretrained_model_vars = get_main_network('inception', input_tensor=img, num_classes=opt.num_class, use_weights=False)
    output = model.output
    logit = tf.cast(tf.argmax(output, axis=1), np.float32)
    # accuracy = tf.reduce_mean(tf.cast(tf.equal(logit, tf.cast(label, tf.float32)), tf.float32))
# define loss
with tf.name_scope('cross_entropy'):
    cross_entropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=output) 
    cross_entropy_loss = tf.reduce_mean(cross_entropy_loss)

''' Initialization '''
init_op = tf.variables_initializer(set(tf.global_variables()) - set(pretrained_model_vars))
sess.run(init_op)

''' Main '''
with sess.as_default():
    try:
        model.load_weights(opt.load_from_checkpoint, by_name=True)
        print ('=> load from checkpoint '+opt.load_from_checkpoint)
    except Exception as e:
        raise ValueError('=> unable to load checkpoint ...' + str(e))
    # do testing 
    test_generator.reset()
    for ti in range(test_iter):
        x_batch, y_batch = next(test_generator)
        # tensorflow wants a different tensor order
        feed_dict = {   
                        img: x_batch,
                        label: y_batch,
                        K.learning_phase(): False # not very sure about it
                    }
        loss, _, pred = sess.run([cross_entropy_loss, output, logit], feed_dict=feed_dict)
        score = vis.add_sample(pred=pred, gt=y_batch)
        if ti % 10 == 0 : 
            print ('TEST [iter %d/%d]: loss=%f, acc=%f' % (ti, test_iter, loss, score))
            sys.stdout.flush()

    vis.compute_scores(suffix=0)