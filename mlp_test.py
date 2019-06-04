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
from utils.visualization import tsne_visualization, auc_evalation, pr_evalation
import json, argparse
from  utils.visualization import AverageMeter

'''Configuration'''
sess = tf.Session()
K.set_session(sess)
tf.set_random_seed(0)

parser = argparse.ArgumentParser(description="Settings for finetuning classifier")
# parser.add_argument('-e', '--execute_mode',  type=str, default='train')
parser.add_argument('--log_dir',  type=str, default='')
parser.add_argument('--feat_root',  type=str, default='')
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
            # for cnn feat
            [[0,1], 4096],
            [[0,1,2,3], 6144],
            # for mdnet feat 4 is semantic knowledge
            [[0,1,4], 4608],
            [[0,1,2,3,4], 6656]
        ]
feat_ids, feat_dim = feat_combinations[args.feat_comb]
if ('mdnet' in args.feat_data_name and 4 not in feat_ids):
    print ('WARNING. mdnet semantic feature is not used')

print ('=> use feature combination', feat_combinations[args.feat_comb])
''' Define data loader'''
test_data_loader = FEATLOADER(batch_size=1, sampling_rate=args.sampling_rate,
        raw_feat_path=os.path.join(args.feat_root, 'seg_test_slides/', args.feat_data_name),
        groundtruth_root='./data/wsi/test_label_293.json',
        shuffle=False, use_selected_slide=False,
        feat_ids=feat_ids, feat_dim=feat_dim) # feat_dim is conditioned on feat_ids
input_dim = test_data_loader.feat_dim

''' Define model '''
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(1024, input_dim=input_dim, activation='relu'))
# model.add(tf.keras.layers.Dropout(args.drop_rate))
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dropout(args.drop_rate))
model.add(tf.keras.layers.Dense(2))

# model.add(tf.keras.layers.Dense(1024, input_dim=input_dim, activation=None))
# model.add(tf.keras.layers.BatchNormalization(axis=-1, scale=False))
# model.add(tf.keras.layers.Lambda(K.relu))
# model.add(tf.keras.layers.Dense(256, activation=None))
# model.add(tf.keras.layers.BatchNormalization(axis=-1, scale=False))
# model.add(tf.keras.layers.Lambda(K.relu))
# model.add(tf.keras.layers.Dropout(args.drop_rate))
# model.add(tf.keras.layers.Dense(2))


fc_layers = [l.output for l in model.layers if 'dense' in l.name]

ignored_variables = []
print ('---------- network ------------ ')
for l in model.layers:
    print (l.name, l.output.shape)

X = model.input
prob_cls = model.output
logits_cls = tf.nn.softmax(prob_cls)

Y = tf.placeholder('int64', name='disease_label')
with tf.name_scope('cnn_optimizer'):
    cross_entropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y, logits=prob_cls)
    loss_op_cls = tf.reduce_mean(cross_entropy_loss)

with sess.as_default():
    sess.run(tf.global_variables_initializer())

    model.load_weights(args.load_model_from+'.h5', by_name=True)

    test_acc_scores = AverageMeter(path=os.path.join(args.log_dir, 'test_acc.json'))
    # global trianing iteration
    print (' --> evaluation ....')
    labels = []
    preds = []
    logits = []
    cnn_preds = []
    cnn_voting_acc = 0.0
    set_layer_outputs = []
    name_list = []
    for s in range(test_data_loader.get_run_num()):
        feat_batch, cls_batch, slide_name = test_data_loader.load_batch_test(duplication=args.duplication)
        name_list.append(slide_name)
        print ('processing {}: {}'.format(s, slide_name))
        feed_dict = {
                        X: feat_batch,
                        Y: cls_batch,
                        K.learning_phase(): 0,
                    }
        cls_logits, test_loss_iter = sess.run([logits_cls, loss_op_cls], feed_dict=feed_dict)

        if args.argmax_predict == True:
            # predict by argmax and voting
            single_pred = np.argmax(cls_logits,1)
            pre_cls = mode(single_pred, axis=0)[0][0]
        else:
            # predict by averaging probs and argmax
            single_logit = np.mean(cls_logits,axis=0)
            pre_cls = np.argmax(single_logit, 0)
        # voting simply
        votes = np.argmax(cls_logits[:,:2],1)
        winner = mode(votes, axis=0)[0][0]
        label = cls_batch[0]
        test_acc = np.sum(np.equal(label, pre_cls))

        if test_acc == 1:
            l1, l2, l3 = sess.run(fc_layers, feed_dict=feed_dict)
            set_layer_outputs.append([feat_batch, l1, l2, l3])

        test_acc_scores.update(test_acc)
        labels += [label]
        preds += [pre_cls]
        logits += [single_logit]
        cnn_preds += [winner]
        cnn_voting_acc += np.sum(np.equal(label, winner))

    # dd.io.save('./experiment/tsne_vis/layer_outputs_tsne.h5', {'layer_outputs': set_layer_outputs}) # validate tsne module
    dd.io.save('./checkpoints/diagnosis_analysis/mlp_{}pred.h5'.format(len(name_list)), {'label': labels, 'logit': np.array(logits), 'name_list': name_list})

    # pr_evalation(labels, np.array(logits), save_path=args.log_dir+"/test_{:0.3f}".format(test_acc_scores.avg))
    # auc_evalation(labels, np.array(logits), save_path=args.log_dir+"/evaluation/test_{:0.3f}".format(test_acc_scores.avg), name_list=name_list, selected_slide=True)
    auc_evalation(labels, np.array(logits))

    conf  = confusion_matrix(labels, preds)
    test_acc = test_acc_scores.avg
    print ('-'*50)
    print ('Overal accuracy (cls): ', test_acc)
    print ('Confusion matrix: ')
    print (conf)
    print ('-'*50)
    conf  = confusion_matrix(labels, cnn_preds)
    cnn_voting_acc = cnn_voting_acc / len(labels)

    print ('Overal accuracy (cnn voting): ', cnn_voting_acc)
    print ('Confusion matrix: ')
    print (conf)
    print ('-'*50)
