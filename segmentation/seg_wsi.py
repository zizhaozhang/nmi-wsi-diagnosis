import os, sys, pdb

from skimage import io, transform, morphology
from skimage import img_as_ubyte
import argparse, shutil, glob, openslide
import numpy as np
import cv2, json
from datetime import datetime
from operator import itemgetter, attrgetter
import scipy.misc as misc

import tensorflow as tf
from tensorflow import keras
K = keras.backend

import util, wsi_util
from wsi_util import patch_sampling, SlideLoader, gradient_merge, visualize_sampling_points, visualize_heatmap
from unet import UNet, preprocess_input
from util import gen_thumbnail, load_model
# from clear_margin_pos import ClearMarginPos
import warnings
warnings.filterwarnings("ignore")

def seg_wsi(args):
    def normalize(data):
        return data / data.max()

    sampled_img_size = 256
    # the border is single-side, e.g. [512,512] -> [528, 528]
    border = 16

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)
    # K.set_learning_phase(0)

    img = tf.placeholder(tf.float32, shape=(None, args.imSize, args.imSize, args.input_channel))
    model = UNet().create_model(img_shape=[opt.imSize, opt.imSize, 3], num_class=opt.num_class, rate=0.0, input_tensor=preprocess_input(img))
    unet_pred = tf.nn.softmax(model.output)

    sess.run(tf.global_variables_initializer())  # initilize

    try:
        model.load_weights(args.load_from_checkpoint)
        print("[*] Success to load model.")
    except:
        sys.exit("[*] Failed to find a checkpoint "+args.load_from_checkpoint)

    print("=> Starting WSI patch generation...")

    wsi_filelist = []
    wsi_filelist.extend(sorted(glob.glob(os.path.join(args.wsi_dir, '*.svs'))))
    wsi_filelist.extend(sorted(glob.glob(os.path.join(args.wsi_dir, '*.tiff'))))
    # wsi_filelist = ['../../dataset/bladder/test_slides_small/104842_sub1_type1.tiff']

    segmented_files = next(os.walk(args.res_dir))[1]

    SlideHandler = SlideLoader(args.batch_size, level=args.slide_level, to_real_scale=4, imsize=args.imSize)
    print("=> Found {} whole slide images in total.".format(len(wsi_filelist)))
    print("=> {} has been processed.".format(len(segmented_files)))
    wsi_filelist = [a for a in wsi_filelist if os.path.splitext(os.path.basename(a))[0] not in segmented_files]
    print("=> {} is being processed.".format(len(wsi_filelist)))
    end = min(args.end, len(wsi_filelist))
    for index in range(args.start, end): # TODO remove this s
        wsi_filepath = wsi_filelist[index]
        wsi_img_name = os.path.splitext(os.path.basename(wsi_filepath))[0]

        if os.path.isdir(os.path.join(args.res_dir, wsi_img_name)) or \
            wsi_img_name in segmented_files:
            continue

        start_time = datetime.now()
        print("=> Start {}/{} segment {}".format(index+1, end, wsi_img_name))

        # if wsi_img_name in ClearMarginPos.keys():
        #     crop_down_scale = int(ClearMarginPos[wsi_img_name][3])
        # else:
        crop_down_scale = 1

        try:
            slide_iterator, num_batches, slide_name, act_slide_size = SlideHandler.get_slide_iterator(wsi_filepath,
                                                                                        down_scale_rate=crop_down_scale,
                                                                                        overlapp=512)
            wsi_seg_results = np.zeros(act_slide_size, dtype=np.float16)
            wsi_img_results = np.zeros(act_slide_size+[3], dtype=np.uint8)
            wsi_mask = np.zeros(act_slide_size, dtype=np.float16) # used to average the overlapping region for border removing
            candidates = []
            with sess.as_default():
                for step, (batch_imgs, locs) in enumerate(slide_iterator):
                    # locs[0]: (y, x)

                    sys.stdout.write('{}-{},'.format(step,(batch_imgs.shape[0])))
                    sys.stdout.flush()
                    feed_dict = {
                        img: batch_imgs,
                        K.learning_phase(): False
                    }
                    batch_pred = sess.run(unet_pred, feed_dict=feed_dict)
                    batch_logits = batch_pred[:,:,:,1]
                    # put the results back to
                    for id, (seg, im, loc) in enumerate(zip(batch_logits, batch_imgs, locs)):
                        y, x = loc[0], loc[1]
                        # there is overlapping
                        seg_h, seg_w = seg.shape
                        # prevent overflow, not happen useually
                        if seg_h+y > wsi_seg_results.shape[0]:
                            y = wsi_seg_results.shape[0] - seg_h
                        if seg_w+x > wsi_seg_results.shape[1]:
                            x = wsi_seg_results.shape[1] - seg_w
                        wsi_mask[y:y+seg_h, x:x+seg_w] = wsi_mask[y:y+seg_h, x:x+seg_w] + 1

                        ## gradient average
                        # diff_mask = wsi_mask[y:y+seg_h, x:x+seg_w].copy()
                        # diff_mask[diff_mask < 2] = 0
                        # wsi_seg_results[y:y+seg_h, x:x+seg_w] = gradient_merge(wsi_seg_results[y:y+seg_h, x:x+seg_w], seg.astype(np.float16), diff_mask)

                        ## simple average
                        # wsi_seg_results[y:y+seg_h, x:x+seg_w] = (wsi_seg_results[y:y+seg_h, x:x+seg_w] + seg.astype(np.float16)) / wsi_mask[y:y+seg_h, x:x+seg_w]

                        ## maximum
                        wsi_seg_results[y:y+seg_h, x:x+seg_w] = np.maximum(wsi_seg_results[y:y+seg_h, x:x+seg_w], seg.astype(np.float16))

                        wsi_img_results[y:y+seg_h, x:x+seg_w] = im.astype(np.uint8)
                        candidates.append([(y, x), seg.copy(), im.copy()])

            # Saving segmentation and sampling results
            cur_dir = os.path.join(args.res_dir, os.path.splitext(wsi_img_name)[0])
            if not os.path.exists(cur_dir):
                # shutil.rmtree(cur_dir)
                os.makedirs(cur_dir)

            # Sample ROI randomly
            sample_patches = patch_sampling(candidates,
                                            tot_samples=args.num_samples,
                                            stride_ratio=0.01,
                                            sample_size=[sampled_img_size,sampled_img_size],
                                            threshold=args.threshold)

            for idx in range(len(sample_patches)):
                file_name_surfix = '_' + str(idx).zfill(5) + '_' + str(sample_patches[idx][0][0]).zfill(6) + \
                                '_' + str(sample_patches[idx][0][1]).zfill(6)
                cur_patch_path = os.path.join(cur_dir, wsi_img_name + file_name_surfix)
                sample_img = (sample_patches[idx][1]).astype(np.uint8)
                sample_seg = (normalize(sample_patches[idx][2])*255.0).astype(np.uint8)
                misc.imsave(cur_patch_path+ '.png', sample_img) # only target images are png format

            locs = [a[0] for a in sample_patches]
            visualize_heatmap((wsi_seg_results*255.0).astype(np.uint8),
                                shape=wsi_img_results.shape, stride=sampled_img_size,
                                wsi_img=wsi_img_results, save_path=os.path.join(cur_dir, wsi_img_name))

            # thumb_img = gen_thumbnail(wsi_img_results, thumb_max=np.max(wsi_seg_results.shape))
            misc.imsave(os.path.join(cur_dir, wsi_img_name + '_seg.jpg'), misc.imresize(wsi_seg_results, 0.5).astype(np.uint8))
            misc.imsave(os.path.join(cur_dir, wsi_img_name + '_thumb.jpg'), misc.imresize(wsi_img_results, 0.5).astype(np.uint8))

            wsi_img_point_results = visualize_sampling_points(wsi_img_results, locs, path=None)
            misc.imsave(os.path.join(cur_dir, wsi_img_name + '_samplepoint.jpg'), misc.imresize(wsi_img_point_results, 0.5).astype(np.uint8))
            elapsed_time = datetime.now() - start_time
            print('=> Time {}'.format(elapsed_time))
        except Exception as e:
            print (e)
            pass

if __name__ == "__main__":
    # args = set_args()
    from opts import *
    if not os.path.isdir(opt.res_dir):
        os.mkdir(opt.res_dir)
    seg_wsi(opt)
