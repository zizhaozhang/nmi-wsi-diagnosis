'''
 * @author [Zizhao Zhang]
 * @email [zizhao@cise.ufl.edu]
 * @create date 2017-05-19 03:06:32
 * @modify date 2017-05-19 03:06:32
 * @desc [description]
'''

SEED=0 # set set to allow reproducing runs
import numpy as np
np.random.seed(SEED)
import tensorflow as tf
tf.set_random_seed(SEED)

import os, sys, time, math
import scipy.misc as misc
from classifier import *
from data_gen import data_loader

from util import VIS
# configure opts
from opts import *

# configuration session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K = tf.keras.backend
K.set_session(sess)

# save and compute metrics
vis = VIS(save_path=opt.checkpoint_path)

''' Users defined data loader (with train and test) '''
img_shape = [opt.imSize, opt.imSize]
train_generator, test_generator, train_samples, test_samples = data_loader(opt.data_path, opt.batch_size, imSize=opt.imSize)


iter_epoch = int(train_samples / opt.batch_size * opt.iter_epoch_ratio)
test_iter = int(test_samples / opt.batch_size)
# define input holders
label = tf.placeholder(tf.int32, shape=[None])
img = tf.placeholder(tf.float32, shape=[None]+img_shape+[3])

# define model
with tf.name_scope('inception'):    
    model, pretrained_model_vars = get_main_network('inception', input_tensor=img, num_classes=opt.num_class, use_weights=opt.use_imagenet_weight)
    output = model.output
    logit = tf.cast(tf.argmax(output, axis=1), np.float32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(logit, tf.cast(label, tf.float32)), tf.float32))

# define loss
with tf.name_scope('cross_entropy'):
    cross_entropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=output)  

    if opt.use_cls_weight > 1:
        print ('=> use class weight',opt.use_cls_weight)
        target_label = 0 # 0 is low grade which has lower amount
        weight = tf.ones(shape=[opt.batch_size])
        cond = tf.equal(label, target_label)
        bweight = tf.where(cond, weight*opt.use_cls_weight, weight) # cross_entrpy/Assign:0
        cross_entropy_loss = cross_entropy_loss * bweight

    cross_entropy_loss = tf.reduce_mean(cross_entropy_loss)
    if opt.weight_decay > 0:
        cross_entropy_loss = cross_entropy_loss + opt.weight_decay * tf.add_n(
                                                [tf.nn.l2_loss(v) for v in tf.trainable_variables()
                                                if 'batch_normalization' not in v.name])
# define optimizer
with tf.name_scope('learning_rate'):
    global_step = tf.Variable(0, name='global_step', trainable=False)
    learning_rate = tf.train.exponential_decay(opt.learning_rate, global_step,
                                        iter_epoch*opt.lr_decay_epoch, opt.lr_decay, staircase=True)
# keras uses tf.layers.BatchNormalization, 
# which requires the following ops to update mean/variance
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    if opt.optim == 'adam':
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    elif opt.optim == 'sgd':
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
    elif opt.optim == 'rmsp':
        optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)  
    train_step = optimizer.minimize(cross_entropy_loss, global_step=global_step)

''' Tensorboard visualization '''
# define summary for tensorboard
with tf.name_scope('summary'):
    tf.summary.scalar('cross_entropy_loss', cross_entropy_loss)
    tf.summary.scalar('learning_rate', learning_rate)
    tf.summary.scalar('accuracy', accuracy)
    summary_merged = tf.summary.merge_all()

''' Initialization '''
tot_iter = iter_epoch * opt.epoch
init_op = tf.variables_initializer(set(tf.global_variables()) - set(pretrained_model_vars))
sess.run(init_op)
# define saver
if not opt.eval:
    train_writer = tf.summary.FileWriter(opt.checkpoint_path+'/train', sess.graph)
    test_writer = tf.summary.FileWriter(opt.checkpoint_path+'/test', sess.graph)

''' Main '''
with sess.as_default():
    if opt.load_from_checkpoint != '':
        try:
            model.load_weights(opt.load_from_checkpoint,by_name=True)
            print ('=> load from checkpoint '+opt.load_from_checkpoint)
        except Exception as e:
            raise ValueError('=> unable to load checkpoint ...' + str(e))
    start = global_step.eval()
    epoch_iter = 0
    epoch_time = 0
    for it in range(start, tot_iter):
        if (it != 0 and it % iter_epoch == 0) or (opt.load_from_checkpoint != '' and it == start) or opt.eval: # or it % 40 == 0:
            test_epoch = it//iter_epoch
            # do testing 
            vis_cands = []
            test_generator.reset()
            for ti in range(test_iter):
                x_batch, y_batch = next(test_generator)
                # tensorflow wants a different tensor order
                feed_dict = {   
                                img: x_batch,
                                label: y_batch,
                                K.learning_phase(): False # not very sure about it
                            }
                loss, pred, acc, w_sum = sess.run([cross_entropy_loss, logit, accuracy, summary_merged], feed_dict=feed_dict)
                if not opt.eval: test_writer.add_summary(w_sum, test_epoch)
                score = vis.add_sample(pred=pred, gt=y_batch)
                if ti % 10 == 0 : 
                    print ('TEST [iter %d/%d, epoch %.3f]: loss=%f, acc=%.3f (%.3f)' % (ti, test_iter, test_epoch, loss, acc, score))
                    sys.stdout.flush()
            overall_acc = vis.compute_scores(suffix=it)

            # save checkpoint
            if it != start:
                model.save_weights(os.path.join(opt.checkpoint_path, 'checkpoint_{}_{:.2f}.h5'.format(test_epoch, overall_acc)))
                print ('=> save checkpoint at '+os.path.join(opt.checkpoint_path, 'checkpoint_{}_{:.2f}.h5'.format(test_epoch, overall_acc)))
                print ('=> epoch time cost:', epoch_time)

            vis.reset()
            epoch_iter = 0
            epoch_time = 0
            
            if opt.eval: break 

        start = time.time()
        x_batch, y_batch  = next(train_generator)
        feed_dict = {   
                        img: x_batch,
                        label: y_batch,
                        K.learning_phase(): True 
                    }
        _, loss, w_sum, lr, pred, score = sess.run([train_step, 
                                    cross_entropy_loss, 
                                    summary_merged,
                                    learning_rate,
                                    logit,
                                    accuracy
                                    ], feed_dict=feed_dict)
        global_step.assign(it).eval()           
        end = time.time() - start
        epoch_time += end
        if it % (iter_epoch//10) == 0: 
            print ('TRAIN [iter %d/%d epoch %.3f time %.3f]: lr=%f loss=%f, acc=%f' % (epoch_iter, iter_epoch, it/iter_epoch, end, lr, loss, score))
            train_writer.add_summary(w_sum, it)
            sys.stdout.flush()
        epoch_iter += 1 