import tensorflow as tf
K = tf.keras.backend

import argparse
import os, pdb
import numpy as np
from feat_loader_inbal import FEATLOADER
from sklearn.metrics import confusion_matrix
from scipy.stats import mode
import sys
import scipy.io as io
import deepdish as dd
from utils.visualization import tsne_visualization, auc_evalation
import json, argparse
from  utils.visualization import AverageMeter

'''Configuration'''
sess = tf.Session()
K.set_session(sess)
tf.set_random_seed(0)

parser = argparse.ArgumentParser(description="Settings for finetuning classifier")
# parser.add_argument('-e', '--execute_mode',  type=str, default='train')
parser.add_argument('--log_dir',  type=str, default='/home/zizhao/work2/mdnet_checkpoints/diagnosis/')
parser.add_argument('--feat_root',  type=str, default='/home/zizhao/work2/mdnet_checkpoints/')
parser.add_argument('--feat_data_name',  type=str, default='')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--load_model_from',  type=str, default='')
parser.add_argument('--tot_epoch', type=int, default=10)
parser.add_argument('--show_iter', type=int, default=10)
parser.add_argument('--test_per_epoch', type=int, default=1)
parser.add_argument('--duplication', type=int, default=10)
parser.add_argument('--learning_rate', type=float, default=0.00001)
parser.add_argument('--lr_decay', type=float, default=0.9)
parser.add_argument('--lr_decay_epoch', type=float, default=1)
parser.add_argument('--drop_rate', type=float, default=0.5)
parser.add_argument('--feat_comb', type=int, default=0)
parser.add_argument('--sampling_rate', type=float, default=0.2)
parser.add_argument('--use_cls_weight', type=int, default=1)
parser.add_argument('--argmax_predict', action='store_true')

args = parser.parse_args()
for arg in vars(args):
    print (arg, getattr(args, arg))

feat_combinations = [
            [[0,1], 4096],
            [[0,1,2,3], 6144],
        ]
feat_ids, feat_dim = feat_combinations[args.feat_comb]

print ('=> use feature combination', feat_combinations[args.feat_comb])
''' Define data loader'''
train_data_loader = FEATLOADER(batch_size=args.batch_size, sampling_rate=args.sampling_rate,
        raw_feat_path=os.path.join(args.feat_root, 'seg_train_slides/', args.feat_data_name),
        groundtruth_root='data/wsi/train_diagnosis.json',
        feat_ids=feat_ids, feat_dim=feat_dim) # feat_dim is conditioned on feat_ids
test_data_loader = FEATLOADER(batch_size=1, sampling_rate=args.sampling_rate,
        raw_feat_path=os.path.join(args.feat_root, 'seg_test_slides/', args.feat_data_name),
        groundtruth_root='data/wsi/val_test_diagnosis.json',
        shuffle=False,
        feat_ids=feat_ids, feat_dim=feat_dim, use_selected_slide=True) # feat_dim is conditioned on feat_ids
input_dim = train_data_loader.feat_dim
iter_epoch = train_data_loader.get_iter_epoch() # how many times increase
tot_iter = iter_epoch * args.tot_epoch

''' Define model '''
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(1024, input_dim=input_dim, activation='relu'))
model.add(tf.keras.layers.Dropout(args.drop_rate))
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dropout(args.drop_rate))
model.add(tf.keras.layers.Dense(2))

ignored_variables = []
print ('---------- network ------------ ')
for l in model.layers:
    print (l.name, l.output.shape)

X = model.input
prob_cls = model.output
logits_cls = tf.nn.softmax(prob_cls)

global_step = tf.Variable(initial_value=0, name="global_step", trainable=False)
Y = tf.placeholder('int64', name='disease_label')

with tf.name_scope('learning_rate'):
    learning_rate = tf.train.exponential_decay(args.learning_rate, global_step, iter_epoch*args.lr_decay_epoch, args.lr_decay, staircase=True)
    tf.summary.scalar('learning_rate', learning_rate)

with tf.name_scope('cnn_optimizer'):
    cross_entropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y, logits=prob_cls)

    if args.use_cls_weight > 1:
        print ('=> use class weight',args.use_cls_weight)
        target_label = 0 # 0 is low grade which has lower amount
        weight = tf.ones_like(cross_entropy_loss)
        cond = tf.equal(Y, target_label)
        bweight = tf.where(cond, weight*args.use_cls_weight, weight) # cross_entrpy/Assign:0
        cross_entropy_loss = cross_entropy_loss * bweight

    loss_op_cls = tf.reduce_mean(cross_entropy_loss)
    lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in tf.trainable_variables()
                        if 'bias' not in v.name or 'batch_normalization' not in v.name]) * 0.0001

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) # for updating batch_norm
    with tf.control_dependencies(update_ops):
        train_op_cnn = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9, use_nesterov=True).minimize(loss_op_cls + lossL2)

with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(Y, tf.argmax(logits_cls, 1) )
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

with tf.name_scope('loss'):
    tf.summary.scalar('loss', loss_op_cls)

merged = tf.summary.merge_all()

def train(train_writer, test_writer):
    with sess.as_default():
        sess.run(tf.global_variables_initializer())
        test_acc_best_list = [0]
        test_acc_scores = AverageMeter(path=os.path.join(args.log_dir, 'test_acc.json'))
        test_acc_loss = AverageMeter(path=os.path.join(args.log_dir, 'test_loss.json'))
        # global trianing iteration
        start = global_step.eval()
        for it in range(start+1, tot_iter):
            # evaluation
            epoch = it // iter_epoch
            ''' Test '''
            if (it / iter_epoch) % args.test_per_epoch == 0:
                print (' --> evaluation ....')
                labels = []
                preds = []
                logits = []
                for s in range(test_data_loader.get_run_num()):
                    feat_batch, cls_batch, name_list = test_data_loader.load_batch_test(duplication=args.duplication)
                    feed_dict = {
                                    X: feat_batch,
                                    Y: cls_batch,
                                    K.learning_phase(): 0,
                                }
                    cls_logits, test_loss_iter, summary = sess.run([logits_cls, loss_op_cls, merged], feed_dict=feed_dict)
                    if args.argmax_predict == True:
                        # predict by argmax and voting
                        single_pred = np.argmax(cls_logits,1)
                        pre_cls = mode(single_pred, axis=0)[0][0]
                        single_logit = np.mean(cls_logits,axis=0)
                    else:
                        # predict by averaging probs and argmax
                        single_logit = np.mean(cls_logits,axis=0)
                        pre_cls = np.argmax(single_logit, 0)

                    label = cls_batch[0]
                    test_acc = np.sum(np.equal(label, pre_cls))
                    test_acc_scores.update(test_acc)
                    test_acc_loss.update(test_loss_iter)
                    labels += [label]
                    preds += [pre_cls]
                    logits += [single_logit]

                # print ('[{}]{}:  label {}, pred {} [vote: {}]'.format(s, name_list, label, pre_cls, single_pred))
                conf = confusion_matrix(labels, preds)
                test_auc = auc_evalation(np.array(labels), np.array(logits))
                print ('-'*50)
                print('test acc [data=%d] = %f loss = %f' % (test_acc_scores.count, test_auc, test_acc_loss.avg) )
                print ('conf matrix')
                print (conf)
                print ('-'*50)



                if test_auc >= test_acc_best_list[-1]:
                    model.save(args.log_dir+'/model_'+str(epoch)+'_'+'%.3f'%(test_auc)+'.h5')
                    test_acc_best_list.append(test_auc)
                    # auc_evalation(labels, np.array(logits), save_path=args.log_dir+"/auc_epoch{}_{:0.3f}_".format(epoch, test_auc))
                    auc_evalation(labels, np.array(logits))

                test_acc_scores.reset_save(epoch)
                test_acc_loss.reset_save(epoch)

            ''' Train '''
            feat_batch, cls_batch, _ = train_data_loader.load_batch()
            nan_c = np.where(np.isnan(feat_batch.reshape(-1)))[0]
            assert(nan_c.size == 0)
            feed_dict = {
                            X: feat_batch,
                            Y: cls_batch,
                            K.learning_phase(): 1,
                        }
            global_step.assign(it).eval()
            cls_logits, loss, _, summary, lr = sess.run([logits_cls, loss_op_cls, train_op_cnn, merged, learning_rate], feed_dict=feed_dict)
            train_writer.add_summary(summary, it)
            auc = auc_evalation(cls_batch, np.array(cls_logits))

            if it % args.show_iter == 0:
                print ('iter=%d(%.3f epoch) lr=%f, loss=%f, train auc=%f' % (it, float(it)/iter_epoch, lr, loss,  auc))
                sys.stdout.flush()
    return conf


# use predefined train and test
train_writer = tf.summary.FileWriter(args.log_dir + '/train')
test_writer = tf.summary.FileWriter(args.log_dir + '/test')
conf = train(train_writer, test_writer)
print(conf)
