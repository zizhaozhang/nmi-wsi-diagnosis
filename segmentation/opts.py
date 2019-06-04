'''
 * @author [Zizhao Zhang]
 * @email [zizhao@cise.ufl.edu]
 * @create date 2017-05-25 02:20:01
 * @modify date 2017-05-25 02:20:01
 * @desc [description]
'''

import argparse
import os

parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', type=int, default=5, help='input batch size')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
parser.add_argument('--lr_decay', type=float, default=0.9, help='learning rate decay')
parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay')
parser.add_argument('--drop_rate', type=float, default=0.0, help='drop rate of unet')

parser.add_argument('--epoch', type=int, default=50, help='# of epochs')
parser.add_argument('--imSize', type=int, default=256, help='then crop to this size')
parser.add_argument('--iter_epoch_ratio', type=float, default=1, help='# of ratio of total images as an epoch')
parser.add_argument('--num_class', type=int, default=2, help='# of classes')
parser.add_argument('--checkpoint_path', type=str, default='', help='where checkpoint saved')
parser.add_argument('--data_path', type=str, default='', help='where dataset saved. See loader.py to know how to organize the dataset folder')
parser.add_argument('--load_from_checkpoint', type=str, default='', help='where checkpoint loaded')
parser.add_argument('--optim', type=str, default='adam', help='which optimizer')
parser.add_argument('--lr_decay_epoch', type=int, default=1, help='how many epoch to decay learning rate')
parser.add_argument('--eval', action='store_true', help='doing evaluation only')

parser.add_argument('--wsi_dir',              type=str, default='')
parser.add_argument('--res_dir',              type=str, default='')
parser.add_argument('--slide_level',          type=int, default=0)
parser.add_argument('--input_channel',        type=int, default=3)
# sample paramters
parser.add_argument('--seg_ratio',            type=int, default=4)
parser.add_argument('--threshold',            type=float, default=0.6)
parser.add_argument('--seg_batch_num',        type=int, default=20)
parser.add_argument('--num_grids',             type=int, default=100)
parser.add_argument('--num_samples',          type=int, default=200)
parser.add_argument('--seed',                 type=int, default=1234)
parser.add_argument('--start',          type=int, default=0)
parser.add_argument('--end',          type=int, default=9999)

opt = parser.parse_args()

args = vars(opt)
print('------------ Options -------------')
for k, v in sorted(args.items()):
    print('%s: %s' % (str(k), str(v)))
print('-------------- End ----------------')

# hardcode here
dataset_mean = [0.5,0.5,0.5]
dataset_std = [0.5,0.5,0.5]

# training data directory
opt.data_path='../data/segmentation/'

# training model directory
checkpoint_root = '../checkpoints/segmentation/'
opt.checkpoint_path = os.path.join(checkpoint_root, opt.checkpoint_path)
if not os.path.isdir(opt.checkpoint_path):
    os.mkdir(opt.checkpoint_path)
if not os.path.isdir(os.path.join(opt.checkpoint_path,'img')):
    os.mkdir(os.path.join(opt.checkpoint_path,'img'))
