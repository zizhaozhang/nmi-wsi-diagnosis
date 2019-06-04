"""
 * @author [Zizhao]
 * @email [zizhao@cise.ufl.edu]
 * @date 2017-03-02 02:06:43
 * @desc [My CVPR paper MDNet]
"""

import tensorflow as tf
K = tf.keras.backend
import os, sys, time
import numpy as np
from termcolor import colored

from utils.topic_data_loader import DataLoader, ParallDataWraper
from utils.evaluation import Evaluation
from classification import classifier
from utils.vocabulary import Vocabulary

from opts import *
from TopicMDNet import TopicMDNet as MDNet

# configuration session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)  # set sess before initialize variable...

## Configure data loader
data_loader_train = DataLoader(FLAGS, which_set='train', image_size=FLAGS.image_size, shuffle=False)
data_loader_val = DataLoader(FLAGS, which_set='test', image_size=FLAGS.image_size, shuffle=False)
p_data_loader_train = ParallDataWraper(data_loader_train, batch_size=FLAGS.batch_size, thread=3)
p_data_loader_val = ParallDataWraper(data_loader_val, batch_size=FLAGS.batch_size, thread=3)

## Configure evaluator
evaluator = Evaluation(FLAGS, data_loader_train)

## Configure the CNN model
img_shape = [FLAGS.image_size, FLAGS.image_size]
feed_img_batch = tf.placeholder(tf.float32, shape=[None]+img_shape+[3])
with tf.variable_scope('PRETRAINED_CNN'):
    print ('--> init the image model ...')
    CNN, CNN_all_vars = classifier.get_main_network('inception', input_tensor=feed_img_batch, num_classes=3, use_weights=False)
    CNN_last_conv = CNN.layers[-3].output

CNN_all_vars = [var for var in tf.trainable_variables() if 'PRETRAINED_CNN' in var.name ]
FLAGS.conv_feat_dim = int(CNN_last_conv.shape[3])
## Configure the language model model
LSTM = MDNet(FLAGS, data_loader_train,
            CNN_last_conv,
            CNN.output, # classifier logit
            CNN.layers[-1].weights[0].op.name,   # the classifier weight op name for get_collection in MDNet
            feed_img_batch=feed_img_batch, # mainly for image summary
            )

def main(_):
    tf.set_random_seed(0)
    (train_txt_pred, train_stop_logits), (test_txt_pred, test_stop) = LSTM.build_model()    # define train and test LSTM

    loss_op_lstm, loss_op_cls, train_op_lstm, train_op_e2e, summary_merged = LSTM.minimize(train_txt_pred, train_stop_logits,
                                                                                CNN.output, FLAGS.init_learning_rate,
                                                                                data_loader_train.get_iter_epoch()*FLAGS.decay_every_iter)
    feed_txt_batch, feed_stop_indictor, feed_label_batch, feed_cls_batch, global_step, cls_acc = LSTM.return_feed_placeholder()

    iter_epoch = data_loader_train.get_iter_epoch()
    tot_iter = iter_epoch * FLAGS.epoch

    sess.run(tf.variables_initializer(
        set(tf.global_variables()) - set([])))
    ''' Saver must put after all variables are defined! '''
    saver = tf.train.Saver(max_to_keep=100)

    with sess.as_default():
        try:
            # need the change the model path
            CNN.load_weights("./checkpoints/classification/trained_model/checkpoint_48_28.77.h5", by_name=True)
            print ('--> load pretrained CNN parameters')
        except:
            print ('--> fail to load CNN parameters')
            return
        # restore from a checkpoint if exists
        if len(FLAGS.load_from_checkpoint) > 0:
            try:
                saver.restore(sess, FLAGS.load_from_checkpoint)
                print ('--> load from checkpoint '+ FLAGS.load_from_checkpoint)
            except Exception as e:
                print (e)
                print ('--> failed to load checkpoint ' + FLAGS.load_from_checkpoint)
                return
        if not FLAGS.test_mode:
            train_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)
        ## ---------------------------------------------------------------------------------------
        start = global_step.eval()
        for it in range(start+1, tot_iter):
            is_end2end = (it//iter_epoch) >= FLAGS.finetune_after
            ''' evaluation '''
            if it % (iter_epoch//2) == 0: # or it % 10 == 0:

                p_data_loader_val.reset()
                val_iter_epoch = data_loader_val.get_iter_epoch()

                for m in range(val_iter_epoch):

                    img_batch, (text_batch, stop_label), cls_batch, gtext_batch, name_batch = p_data_loader_val.load_batch()
                    if m%10==0: print ('-->{}/{} evaluating [batch_size={}] ...'.format(m, val_iter_epoch, img_batch.shape[0]), flush=True)
                    # sys.stdout.flush()
                    feed_dict = {
                                    feed_img_batch: img_batch,
                                    feed_txt_batch: text_batch[:,:-1], # only use the first feature indictor
                                    feed_stop_indictor: stop_label, # not used
                                    K.learning_phase(): False,
                                    LSTM.is_train_mode: False,
                                    LSTM.feed_keep_drop: 1,
                                }
                    txt_predict_logits, txt_predict_stops, cls_logits, att_probs, cam_att = sess.run([test_txt_pred, test_stop, CNN.output, LSTM.att_probs, LSTM.cam_att], feed_dict=feed_dict)
                    # acc = np.mean(np.equal(cls_batch, np.argmax(cls_logits,1)))
                    evaluator.data_loader.add_stops(txt_predict_stops)
                    evaluator.add_accuracy(txt_predict_logits, cls_logits, text_batch[:,1:], cls_batch,
                                           name_batch, gtext_batch, img_batch, att_probs, cam_att,
                                           verbose=FLAGS.test_mode)

                iter_vbs = '%.1f' % (float(it)/iter_epoch)
                best_verbose = evaluator.summary_overall_evaluation(os.path.join(FLAGS.log_dir,'test_result_'+iter_vbs),
                                                                    save_to_mat=False,
                                                                    single_vis=False,
                                                                    no_eval=FLAGS.test_mode,
                                                                    test_groundtruth_json='./metric/test_annotation_striped_topicmdnet.json')

                ''' save checkpoint '''
                if it > start + 10: # add a smaller number to prevent overwrite the checkpoint when loading back.
                    ckpt_p = os.path.join(FLAGS.log_dir, 'model'+iter_vbs+best_verbose)
                    saver.save(sess, ckpt_p, global_step=global_step)
                    print ("--> save a checkpoint at " + ckpt_p, flush=True)

            ''' training '''
            op_time = time.time()
            img_batch, (text_batch, stop_label), cls_batch, gtext_batch, _ = p_data_loader_train.load_batch()
            feed_dict = {
                            feed_img_batch: img_batch,
                            feed_txt_batch: text_batch[:,:-1],
                            feed_cls_batch: cls_batch,
                            feed_label_batch: text_batch,
                            feed_stop_indictor: stop_label,
                            K.learning_phase(): True,
                            LSTM.is_train_mode: True,
                            LSTM.feed_keep_drop: 0.6,
                        }
            if not is_end2end:
                # training only LSTM
                ops = [train_op_lstm, loss_op_lstm, loss_op_cls, summary_merged, train_txt_pred]
            else:
                # train end to end
                ops = [train_op_e2e, loss_op_lstm, loss_op_cls, summary_merged, train_txt_pred]

            _, loss_lstm, loss_cls, summary, txt_preds = sess.run(ops, feed_dict=feed_dict)


            global_step.assign(it).eval()
            train_writer.add_summary(summary, it)
            if it % 10 == 0:
                print('[epoch %.4f | iter=%d | time=%.3f]: loss=(lstm=%.3f, cls=%.3f), e2e=%s' % (it/iter_epoch, it, time.time()-op_time, loss_lstm, loss_cls, is_end2end), flush=True)

if __name__ == "__main__":
    tf.app.run()
