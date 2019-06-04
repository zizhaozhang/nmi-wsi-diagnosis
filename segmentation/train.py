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

import os, shutil, sys, time, math
import scipy.misc as misc
from unet import UNet, preprocess_input
from data_gen import data_loader

from util import VIS, mean_IU, vis_overlay, make_grid, dice_coef, VISRecall
# configure opts
from opts import *
from opts import dataset_mean, dataset_std # set them in opts
K = tf.keras.backend
# save and compute metrics
vis = VISRecall(save_path=opt.checkpoint_path)

# configuration session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

''' Users defined data loader (with train and test) '''
img_shape = [opt.imSize, opt.imSize]
train_generator, test_generator, train_samples, test_samples = data_loader(opt.data_path, opt.batch_size, imSize=opt.imSize, mean=dataset_mean, std=dataset_std)

iter_epoch = int(train_samples / opt.batch_size * opt.iter_epoch_ratio)
test_iter = int(test_samples / opt.batch_size)

# define input holders
label = tf.placeholder(tf.int32, shape=[None]+img_shape)
label_weights = tf.placeholder(tf.float32, shape=[None]+img_shape)
is_training = tf.placeholder(tf.bool, name='training_mode_placeholder')

# define model
with tf.name_scope('unet'):
    img = tf.placeholder(tf.float32, shape=(None, opt.imSize, opt.imSize, 3))
    model = UNet().create_model(img_shape=img_shape+[3], num_class=opt.num_class, rate=opt.drop_rate, input_tensor=preprocess_input(img))
    pred = model.output
    logit = tf.nn.softmax(pred) # used in test
    # logit = tf.cast(tf.argmax(pred, axis=3), np.float32)
# define loss
with tf.name_scope('cross_entropy'):
    cross_entropy_loss_pixel = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=pred)
    cross_entropy_loss_pixel = tf.multiply(cross_entropy_loss_pixel, label_weights)
    cross_entropy_loss = tf.reduce_sum(cross_entropy_loss_pixel) / (tf.reduce_sum(label_weights) + 0.00001)
    if opt.weight_decay > 0:
        cross_entropy_loss = cross_entropy_loss + opt.weight_decay * tf.add_n(
                                                [tf.nn.l2_loss(v) for v in tf.trainable_variables()
                                                if 'batch_normalization' not in v.name])
# define optimizer
global_step = tf.Variable(0, name='global_step', trainable=False)
with tf.name_scope('learning_rate'):
    learning_rate = tf.train.exponential_decay(opt.learning_rate, global_step,
                                           iter_epoch*opt.lr_decay_epoch, opt.lr_decay, staircase=True)
    if opt.optim == 'adam':
        train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy_loss, global_step=global_step)
    elif opt.optim == 'sgd':
        train_step = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9, use_nesterov=True).minimize(cross_entropy_loss, global_step=global_step)

''' Tensorboard visualization '''
# define summary for tensorboard
with tf.name_scope('summary'):
    tf.summary.scalar('cross_entropy_loss', cross_entropy_loss)
    tf.summary.scalar('learning_rate', learning_rate)
    # tf.summary.image('input', img, max_outputs=3)
    # tf.summary.image('label', tf.expand_dims(tf.cast(label, tf.float32),3), max_outputs=3)
    # tf.summary.image('weight', tf.expand_dims(label_weights, 3), max_outputs=3)
    # tf.summary.image('prediction', tf.expand_dims(logit, 3), max_outputs=3)
    summary_merged = tf.summary.merge_all()

# define saver
if not opt.eval:
    train_writer = tf.summary.FileWriter(opt.checkpoint_path+'/train', sess.graph)
    test_writer = tf.summary.FileWriter(opt.checkpoint_path+'/test', sess.graph)
    saver = tf.train.Saver(max_to_keep=20) # must be added in the end
else:
    saver = tf.train.Saver(var_list=tf.trainable_variables()+[a for a in tf.global_variables() if 'step' in a.name], max_to_keep=20)

''' Main '''
tot_iter = iter_epoch * opt.epoch
init_op = tf.global_variables_initializer()
sess.run(init_op)
with sess.as_default():
    # restore from a checkpoint if exists
    # the name_scope can not change
    if opt.eval:
        try:
            model.load_weights(opt.load_from_checkpoint, by_name=True)
            print ('=> load from checkpoint '+opt.load_from_checkpoint)
        except Exception as e:
            raise ValueError('=> unable to load checkpoint ...' + str(e))
    start = global_step.eval()
    epoch_iter = 0
    for it in range(start, tot_iter):
        if it % iter_epoch == 0 and it != start or opt.eval:
            test_epoch = it//iter_epoch
            # save checkpoint
            if it != start:
                model.save_weights(os.path.join(opt.checkpoint_path, 'checkpoint_{}.h5'.format(test_epoch)))
                print ('=> save checkpoint at '+os.path.join(opt.checkpoint_path, 'checkpoint_{}.h5'.format(test_epoch)))
            # do testing
            vis_cands = []
            for ti in range(test_iter):
                x_batch, y_batch, weight_batch, _ = next(test_generator)

                feed_dict = {
                                img: x_batch,
                                label: y_batch,
                                label_weights: weight_batch,
                                K.learning_phase(): False # not very sure about it
                            }
                loss, pred_logits, w_sum = sess.run([cross_entropy_loss, logit, summary_merged], feed_dict=feed_dict)
                if not opt.eval: test_writer.add_summary(w_sum, test_epoch)
                # pred_map_batch = np.argmax(pred_logits, axis=3)
                pred_map_batch = pred_logits[:,:,:,1] # positive class logits
                for pred_map, y, wmask in zip(pred_map_batch, y_batch, weight_batch):
                    score = vis.add_sample(pred_map, y, wmask)

                p = np.random.randint(0, x_batch.shape[0]-1)
                res = vis_overlay(x_batch, y_batch, pred_map_batch>0.5) # visualize the first in the batch
                vis_cands.append(res)

                if ti % 10 == 0 :
                    print ('TEST [iter %d/%d, epoch %.3f]: loss=%f, mean_IU=%f' % (ti, test_iter, test_epoch, loss, score))
                    sys.stdout.flush()

            vis.compute_scores(suffix=it)
            tot_save = 0
            if opt.eval:
                tot_save = len(vis_cands[0])
            for tt in range(tot_save):
                save_list = [a[tt] for a in vis_cands]
                vis_grid = make_grid(save_list, ncols=6)
                vis_grid = misc.imresize(vis_grid, 0.5)
                misc.imsave(os.path.join(opt.checkpoint_path, 'img/test_results_{}_{}.png'.format(str(test_epoch), str(tt))), vis_grid)
            vis.reset()
            epoch_iter = 0
            if opt.eval: break

        start = time.time()
        x_batch, y_batch, weight_batch, _  = next(train_generator)
        feed_dict = {
                        img: x_batch,
                        label: y_batch,
                        label_weights: weight_batch,
                        K.learning_phase(): True
                    }
        _, loss, w_sum, lr, pred_logits = sess.run([train_step,
                                    cross_entropy_loss,
                                    summary_merged,
                                    learning_rate,
                                    pred
                                    ], feed_dict=feed_dict)
        global_step.assign(it).eval()
        end = time.time() - start
        pred_map = np.argmax(pred_logits[0], axis=2)
        score, _ = mean_IU(pred_map, y_batch[0])
        if it % 20 == 0:
            print ('TRAIN [iter %d/%d epoch %.3f time %.3f]: lr=%f loss=%f, mean_IU=%f' % (epoch_iter, iter_epoch, it/iter_epoch, end, lr, loss, score))
            train_writer.add_summary(w_sum, it)
            sys.stdout.flush()
        epoch_iter += 1
