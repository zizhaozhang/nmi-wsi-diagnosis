
import tensorflow as tf
import os


root = os.path.expanduser('~')+'/work/tf_tmp_checkpoints/'

FLAGS = tf.app.flags.FLAGS
# general parameters
tf.flags.DEFINE_integer("batch_size", 20,
                       "batch size")
tf.flags.DEFINE_string("log_dir", root+'checkpoint_mdnet',
                       "log_dir")
tf.flags.DEFINE_float("init_learning_rate", 0.001,
                       "learning rate")
tf.flags.DEFINE_float("lr_decay_rate", 0.9,
                       "learning rate decay rate")
# tf.flags.DEFINE_float("cnn_learning_rate", 0.001,
#                        "learning rate")
tf.flags.DEFINE_integer("epoch", 10, "Number of training epochs.")
tf.flags.DEFINE_integer("log_every_epoch", 1,
                        "Frequency at which loss and global step are logged.")
tf.flags.DEFINE_string("load_from_checkpoint", '',
                       "if load checkpoint")
tf.flags.DEFINE_string("dataset_dir", "",
                       "dataset directory")
tf.flags.DEFINE_integer("decay_every_iter", 2, "decrease learning rate every .. ")
tf.flags.DEFINE_float("att_weight", 0.9, "attention merge weigth")
## CNN param
tf.flags.DEFINE_integer("conv_feat_wh", 6, "Feature map width.")
tf.flags.DEFINE_integer("conv_feat_dim", 512, "Number of feature map of CNN outputs.")
tf.flags.DEFINE_integer("image_size", 256, "Input image size.")
tf.flags.DEFINE_integer("finetune_after", 3, "after how many epoch to fine tune CNN")
tf.flags.DEFINE_float("cls_loss_weight", 0.1, "attention merge weigth")

# tf.flags.DEFINE_bool("freeze_all", False, "if freeze all layers of CNN")
# tf.flags.DEFINE_bool("end2end", True, "if backpropa LSTM loss to CNN")

tf.flags.DEFINE_bool("test_mode", False, "if just do evaluation on test set")
# tf.flags.DEFINE_string("split", 'test', "which set is used to validation")
tf.flags.DEFINE_bool("sample_max", True, "sample_max for rnn sampling")

tf.flags.DEFINE_string("test_image_dir", "", "dataset directory")
tf.flags.DEFINE_string("model", "", "which model to config")

tf.flags.DEFINE_string("extract_whichset", "train", "extract feature set")

tf.flags.DEFINE_bool("to_demo_input", False, "save files to show demo")


tf.logging.set_verbosity(tf.logging.INFO)
