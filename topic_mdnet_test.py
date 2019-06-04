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
data_loader_val = DataLoader(FLAGS, which_set='train', image_size=FLAGS.image_size, shuffle=False, use_augmentation=False)
p_data_loader_val = ParallDataWraper(data_loader_val, batch_size=FLAGS.batch_size, thread=3)

## Configure evaluator
evaluator = Evaluation(FLAGS, data_loader_val)

## Configure the CNN model
feed_img_batch = tf.placeholder(tf.float32, shape=[None, FLAGS.image_size, FLAGS.image_size, 3])
with tf.variable_scope('PRETRAINED_CNN'):
    print ('--> init the image model ...')
    CNN, CNN_all_vars = classifier.get_main_network('inception', input_tensor=feed_img_batch, num_classes=3, use_weights=False)
    CNN_last_conv = CNN.layers[-3].output

CNN_all_vars = [var for var in tf.trainable_variables() if 'PRETRAINED_CNN' in var.name ]
FLAGS.conv_feat_dim = int(CNN_last_conv.shape[3])
## Configure the language model model
LSTM = MDNet(FLAGS, data_loader_val,
            CNN_last_conv,
            CNN.output, # classifier logit
            CNN.layers[-1].weights[0].op.name,   # the classifier weight op name for get_collection in MDNet
            feed_img_batch=feed_img_batch, # mainly for image summary
            )

def main(_):
    tf.set_random_seed(0)
    _, (test_txt_pred, test_stop) = LSTM.build_model()    # define train and test LSTM

    sess.run(tf.variables_initializer(
        set(tf.global_variables()) - set([])))
    ''' Saver must put after all variables are defined! '''
    saver = tf.train.Saver()

    with sess.as_default():
        # restore from a checkpoint if exists
        try:
            saver.restore(sess, FLAGS.load_from_checkpoint)
            print ('--> load from checkpoint '+ FLAGS.load_from_checkpoint)
        except Exception as e:
            print (e)
            print ('--> failed to load checkpoint ' + FLAGS.load_from_checkpoint)
            sys.exit(0)

        ## ---------------------------------------------------------------------------------------
        ''' evaluation '''
        p_data_loader_val.reset()
        val_iter_epoch = data_loader_val.get_iter_epoch()
        # val_iter_epoch = 10
        for m in range(val_iter_epoch):

            img_batch, (text_batch, stop_label), cls_batch, gtext_batch, name_batch = p_data_loader_val.load_batch()
            if m%10==0: print ('-->{}/{} evaluating [batch_size={}] ...'.format(m, val_iter_epoch, img_batch.shape[0]), flush=True)
            feed_dict = {
                            feed_img_batch: img_batch,
                            LSTM.feed_txt_batch: text_batch[:,:-1], # only use the first feature indictor
                            LSTM.feed_stop_indictor: stop_label, # not used
                            K.learning_phase(): 0,
                            LSTM.is_train_mode: False,
                            LSTM.feed_keep_drop: 1,
                        }
            txt_predict_logits, txt_predict_stops, cls_logits, att_probs, cam_att = sess.run([test_txt_pred, test_stop, CNN.output, LSTM.att_probs,LSTM.cam_att], feed_dict=feed_dict)
            # acc = np.mean(np.equal(cls_batch, np.argmax(cls_logits,1)))
            evaluator.data_loader.add_stops(txt_predict_stops)
            evaluator.add_accuracy(txt_predict_logits, cls_logits, text_batch[:,1:], cls_batch,
                                    name_batch, gtext_batch, img_batch, att_probs, cam_att,
                                    verbose=FLAGS.test_mode)

        best_verbose = evaluator.summary_overall_evaluation(FLAGS.load_from_checkpoint,
                                        save_to_mat=False,
                                        single_vis=True,
                                        no_eval=FLAGS.test_mode,
                                        test_groundtruth_json='./metric/test_annotation_striped_topicmdnet.json')


if __name__ == "__main__":
    tf.app.run()
