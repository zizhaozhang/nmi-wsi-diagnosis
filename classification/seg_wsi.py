import os, sys, pdb
import tensorflow as tf
# Adding local Keras
HOME_DIR = os.path.expanduser('~')
keras_version = 'keras_pingpong'
KERAS_PATH = os.path.join(HOME_DIR, 'Github', keras_version)
sys.path.insert(0, KERAS_PATH)
sys.path.insert(0, os.path.join(KERAS_PATH, 'keras'))
sys.path.insert(0, os.path.join(KERAS_PATH, 'keras', 'layers'))
import keras
from keras import backend as K

from skimage import io, transform, morphology
from skimage import img_as_ubyte
import argparse, shutil, glob, openslide
import matplotlib.pyplot as plt
import numpy as np
import cv2
from datetime import datetime
from operator import itemgetter, attrgetter

import unet, util, wsi_util
from wsi_util import split_wsi_patch_border, patch_sampling
from unet import unet_model, save_model, load_model, dice_coef
from util import gen_thumbnail
from clear_margin_pos import ClearMarginPos
from data_gen import norm_mean_img

def set_args():
    parser = argparse.ArgumentParser(description="Settings for Unet WSI Segmentation")
    # setting paramters
    parser.add_argument('-g', '--gpu_serial_num', type=str, default='0')
    # image parameters
    parser.add_argument('--checkpoint_dir',       type=str, default='./Checkpoints')
    parser.add_argument('--wsi_dir',              type=str, default='/media/pingjun/DataArchiveZizhao/merged/Segmentation/extra1/')
    parser.add_argument('--res_dir',              type=str, default='/media/pingjun/DataArchiveZizhao/merged/Segmentation/Extra1Results/')
    parser.add_argument('--slide_level',          type=int, default=0)
    parser.add_argument('--input_channel',        type=int, default=3)
    # sample paramters
    parser.add_argument('--seg_ratio',            type=int, default=4)
    parser.add_argument('--threshold',            type=float, default=0.6)
    parser.add_argument('--seg_batch_num',        type=int, default=20)
    parser.add_argument('--num_grids',             type=int, default=100)
    parser.add_argument('--num_samples',          type=int, default=200)
    parser.add_argument('--sample_rows',          type=int, default=500)
    parser.add_argument('--sample_cols',          type=int, default=500)
    parser.add_argument('--seed',                 type=int, default=1234)

    args = parser.parse_args()
    return args

def seg_wsi(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_serial_num
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)

    img = tf.placeholder(tf.float32, shape=(None, None, None, args.input_channel))
    unet_pred = unet_model(inputs=img)

    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())  # initilize
    if load_model(sess, saver, args.checkpoint_dir):
        print(" [*] Success to Load model.")
    else:
        sys.exit(" [*] Failed to find a checkpoint")

    K.set_learning_phase(0)
    print("Starting WSI patch generation...")

    wsi_filelist = []
    wsi_filelist.extend(glob.glob(os.path.join(args.wsi_dir, '*.svs')))
    wsi_filelist.extend(glob.glob(os.path.join(args.wsi_dir, '*.tiff')))

    segmented_files = next(os.walk(args.res_dir))[1]

    print("There are {} whole slide images in total.".format(len(wsi_filelist)))
    for index, wsi_filepath in enumerate(wsi_filelist):
        print("Segment {}/{}".format(index+1, len(wsi_filelist)))
        wsi_img_name = os.path.splitext(os.path.basename(wsi_filepath))[0]

        if wsi_img_name in segmented_files: # if already segmented, continue
            continue

        start_time = datetime.now()
        print("---Start segment {}".format(wsi_img_name))

        if wsi_img_name in ClearMarginPos.keys():
            crop_height_range = ClearMarginPos[wsi_img_name][1]
            crop_width_range = ClearMarginPos[wsi_img_name][2]
            crop_scale = int(ClearMarginPos[wsi_img_name][3])
        else:
            crop_height_range = (0.0, 1.0)
            crop_width_range = (0.0, 1.0)
            crop_scale = 2

        # Calculate height/width start/end information in WSI
        slide_img = openslide.open_slide(wsi_filepath)
        SlideWidth, SlideHeight = slide_img.level_dimensions[args.slide_level]
        print("---Image height/width is {}/{}".format(SlideHeight, SlideWidth))
        slide_width_start = int(np.floor(crop_width_range[0] * SlideWidth))
        slide_width_end = int(np.ceil(crop_width_range[1] * SlideWidth))
        crop_slide_width = slide_width_end - slide_width_start
        slide_height_start = int(np.floor(crop_height_range[0] * SlideHeight))
        slide_height_end = int(np.ceil(crop_height_range[1] * SlideHeight))
        crop_slide_height = slide_height_end - slide_height_start

        crop_seg = np.zeros((crop_slide_height/args.seg_ratio, crop_slide_width/args.seg_ratio)) # take large memory
        level_scale = [a/b for (a,b) in zip(slide_img.level_dimensions[0], slide_img.level_dimensions[args.slide_level])]

        if crop_scale == 1:
            seg_mag = 4
            sample_size = [args.sample_rows * 2, args.sample_cols * 2]
        elif crop_scale == 2:
            seg_mag = 2
            sample_size = [args.sample_rows, args.sample_cols]
        else:
            sys.exit("Undeclared crop_scale parameters...")
        patch_width, patch_height, border_size = 512*seg_mag, 512*seg_mag, 48*seg_mag

        # todo: can be probamatic in small images, should be working on image larger than 3000*3000
        seg_width, seg_height = patch_width+2*border_size, patch_height+2*border_size
        split_coords = split_wsi_patch_border(slide_width_start, slide_width_end, slide_height_start, slide_height_end,
                                              patch_width=patch_width, patch_height=patch_height, border_size=border_size)

        print('---There are {} regions.'.format(len(split_coords)))
        batch_num = len(split_coords) / args.seg_batch_num
        last_num = len(split_coords) % args.seg_batch_num

        img_ranges = []
        for ibatch in range(batch_num):
            img_ranges.append((ibatch*args.seg_batch_num, (ibatch+1)*args.seg_batch_num))
        if last_num > 0:
            img_ranges.append((len(split_coords)-last_num, len(split_coords)))

        img_list = []
        read_no_err = 1
        # Batch processing all patches
        for ibatch, img_range in enumerate(img_ranges):
            sys.stdout.write("\rBatch {}/{}".format(ibatch+1, len(img_ranges)))
            sys.stdout.flush()

            batch_imgs, ori_imgs = [], []
            for coord in split_coords[img_range[0]:img_range[1]]:
                seg_coord = coord[1]
                try: # check read region, if error, skip this image
                    cur_patch = slide_img.read_region((seg_coord[0]*level_scale[0], seg_coord[1]*level_scale[1]),
                                                      args.slide_level, (seg_width, seg_height))
                except:
                    print("Read region error:")
                    read_no_err = 0
                    break

                cur_patch = np.asarray(cur_patch)[:,:,0:-1]
                ori_imgs.append(cur_patch)
                cur_patch = transform.resize(cur_patch, (seg_height/seg_mag, seg_width/seg_mag))
                batch_imgs.append(cur_patch)

            if read_no_err == 0: # break second loop: read error
                break
            batch_imgs = np.array(batch_imgs)
            batch_segs = np.squeeze(unet.sigmoid(sess.run(unet_pred, feed_dict={img: batch_imgs})), axis=3)

            # merge segmentation results
            for idx, coord in enumerate(split_coords[img_range[0]:img_range[1]]):
                actual_coord, seg_coord = coord[0], coord[1]
                # saving segmentation results with certain ratio
                height_start = (actual_coord[1]-slide_height_start) / args.seg_ratio
                height_end = (actual_coord[1]-slide_height_start+patch_height) / args.seg_ratio
                width_start = (actual_coord[0]-slide_width_start) / args.seg_ratio
                width_end = (actual_coord[0]-slide_width_start+patch_width) / args.seg_ratio
                start_h = (actual_coord[1]-seg_coord[1]) / args.seg_ratio
                start_w = (actual_coord[0]-seg_coord[0]) / args.seg_ratio

                seg_result = transform.resize(batch_segs[idx,...], (seg_height/args.seg_ratio, seg_width/args.seg_ratio))
                seg_result = seg_result[start_h:start_h+patch_height/args.seg_ratio, start_w:start_w+patch_width/args.seg_ratio]
                crop_seg[height_start:height_end, width_start:width_end] = seg_result

                # saving img_list for sampling
                start_h = (actual_coord[1]-seg_coord[1])
                start_w = (actual_coord[0]-seg_coord[0])
                seg_result = transform.resize(batch_segs[idx,...], (seg_height, seg_width))
                seg_result = seg_result[start_h:start_h+patch_height, start_w:start_w+patch_width]

                seg_priority = np.count_nonzero(seg_result > args.threshold)*1.0 / np.prod(seg_result.shape)
                ori_img = ori_imgs[idx][start_h:start_h+patch_height, start_w:start_w+patch_width]
                img_list.append([seg_priority, (actual_coord, seg_result, ori_img)])

            # remove grids with few segmentation areas
            if ((ibatch + 1) * args.seg_batch_num) %  (args.num_grids * 2) == 0 or (ibatch+1) == len(img_ranges):
                img_list.sort(key=itemgetter(0), reverse=True)
                if len(img_list) > args.num_grids:
                    del img_list[args.num_grids:]

        if read_no_err == 0: # read_err stop current image
            continue

        # Saving segmentation and sampling results
        wsi_img_name = os.path.splitext(wsi_img_name)[0]
        # cur_dir = os.path.join(os.getcwd(), 'ResultsTCGA', wsi_img_name)
        cur_dir = os.path.join(args.res_dir, wsi_img_name)
        if os.path.exists(cur_dir):
            shutil.rmtree(cur_dir)
        os.makedirs(cur_dir)

        sample_patches = patch_sampling(zip(*img_list)[1], tot_samples=args.num_samples, stride_ratio=0.03,
                                        sample_size=sample_size, threshold=args.threshold)
        for idx in range(len(sample_patches)):
            file_name_surfix = '_' + str(idx).zfill(5) + '_' + str(sample_patches[idx][0][0]).zfill(6) + \
                '_' + str(sample_patches[idx][0][1]).zfill(6) + '.png'
            cur_patch_path = os.path.join(cur_dir, wsi_img_name + file_name_surfix)
            if crop_scale == 1:
                cur_sample_patch = (transform.resize(sample_patches[idx][1], (args.sample_rows, args.sample_cols)) * 255).astype(np.uint8)
            elif crop_scale == 2:
                cur_sample_patch = sample_patches[idx][1]
            else:
                sys.exit("Undeclared crop_scale parameters...")
            io.imsave(cur_patch_path, cur_sample_patch)

        seg_name = wsi_img_name + '_seg.jpg'
        cv2.imwrite(os.path.join(cur_dir, seg_name), (crop_seg*255.0).astype(np.uint8))
        thumb_name = wsi_img_name + '_thumb.jpg'
        thumb_img = gen_thumbnail(crop_seg)
        cv2.imwrite(os.path.join(cur_dir, thumb_name), (thumb_img*255.0).astype(np.uint8))

        elapsed_time = datetime.now() - start_time
        print('---Takes {}'.format(elapsed_time))

if __name__ == "__main__":
    args = set_args()
    seg_wsi(args)
