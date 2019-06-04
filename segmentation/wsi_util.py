import os, sys, pdb
import numpy as np
import random

import os, sys
import numpy as np
from io import BytesIO
from PIL import Image
import math
import cv2
import scipy.misc as misc
import deepdish as dd
import openslide, pickle, skimage
from skimage import transform
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def check_kfb(filepath):
    try:
        assert os.path.exists(filepath), "Slide path doesnot exist"
        slide = openslide.open_slide(filepath)
        slide_width, slide_height = slide.level_dimensions[0]
        slide_name = os.path.basename(filepath)
        print("Slide {} original width: {}, height: {}".format(slide_name, slide_width, slide_height))

        return True
    except Exception as e:
        print (e)
        return False

class SlideLoader():
    def __init__(self, batch_size, to_real_scale=4, level=1, imsize=512):

        # assert level ==0, 'level other than 0 is not supported'
        # The acutally size for each sample is imsize / scale, 
        # then resized to imsize for network input
        self.imsize = imsize
        self.batch_size = batch_size
        self.level = level
        self.scale = 1.0 / to_real_scale
        
    def iterator(self, slide, loc):
        idx = 0
        for i in range(0, len(loc)):
            imsize = int((self.actual_crop_size) * self.scale)
            batch_img = np.zeros((self.batch_size, imsize, imsize, 3), dtype=np.float16)
            cur_left = len(loc) - idx
            if cur_left == 0: break
            act_batch_size = min(cur_left, self.batch_size)
            all_loc = [] # loc after scaled down. It is used to stich the patches back
            for b in range(act_batch_size):
                y, x = loc[idx] 
                all_loc.append((int(y*self.scale),int(x*self.scale)))
                origin_img = slide.read_region((x,y), self.level, (self.actual_crop_size, self.actual_crop_size))
                origin_img = np.asarray(origin_img)[:,:,0:-1] # remove the alpha channel
                origin_img = misc.imresize(origin_img, self.scale, interp='nearest')

                batch_img[b] = origin_img
                idx += 1
            yield batch_img[:act_batch_size], all_loc
                    
    def get_slide_iterator(self, path, down_scale_rate, overlapp=128):
        # start_pos: (y, x)
        name = os.path.basename(path)
        name = os.path.splitext(name)[0]

        if check_kfb(path):

            slide_pointer = openslide.open_slide(path)
            w, h = slide_pointer.level_dimensions[self.level]
            # recalculate 
            if down_scale_rate > 1:
                level = 0
            else:
                level = self.level
            self.level_ratio = slide_pointer.level_dimensions[0][0] // slide_pointer.level_dimensions[level][0]
            self.scale = self.scale * self.level_ratio
            self.actual_crop_size = int(self.imsize / self.scale)

            n_width = w // (self.actual_crop_size - overlapp)
            n_height = h // (self.actual_crop_size - overlapp)

            out_w = w * self.scale #int(n_width * self.actual_crop_size * self.scale)
            out_h = h * self.scale #int(n_height * self.actual_crop_size * self.scale)
            
            # compute location index
            loc = []
            
            for i in range(n_height+2):
                for j in range(n_width+2):
                    y = i * (self.actual_crop_size - overlapp)
                    x = j * (self.actual_crop_size - overlapp)
                    y = max(y, 0)
                    x = max(x, 0)
                    if y + self.actual_crop_size > h:
                        y = h - self.actual_crop_size
                    if x + self.actual_crop_size > w:
                        x = w - self.actual_crop_size
                        
                    loc.append((y, x))
            print ('Iterator: ({}, {}) at level {} (ratio {}, actual_size {} (overlap {}))'.format(h, w, self.level, self.level_ratio, self.actual_crop_size, overlapp))
            num_batches = len(loc) // self.batch_size 

            print ('{} patches, splited to {} batches'.format(len(loc), num_batches ))
            
            return self.iterator(slide_pointer, loc), num_batches, name, [int(out_h), int(out_w)]
                
        else:
            raise ValueError('can not load slide {}'.format(path))

def compute_grid_score(grid, threshold):
    grid[grid < threshold] = 0
    # return float(np.count_nonzero(grid))
    return float(grid.sum())

def patch_sampling(img_list, tot_samples=200, stride_ratio=0.1, sample_size=[256,256], threshold=0.5):
    '''
        Input:
            img_list: [((x,y), seg, img), ...]
                    seg: a probability segmentation from
                    (x, y) is the absolute coordinates respect to the whole slide
                    img: is the corresponding image of seg
            sample_size: sample patch size
            stride_ratio: the sampling stride ratio. high value, sparser sampling rate
            threshold: the segmentation prob. belown this value will not be sampled

        Output:
            samples: [((x,y), sampled_img_patch), ...]
                     (x, y) is the absolute coordinates respect to the whole slide
    '''

    print('patch_sampling', '{} patches'.format(len(img_list)))

    # compute sample_mask
    assert(sample_size[0]%2 == 0)
    spl_h_stride, spl_w_stride = sample_size[0]//2, sample_size[1]//2
    img_h, img_w = img_list[0][2].shape[0:2]
    sample_mask = np.zeros((img_h, img_w), np.float32)
    stride_h, stride_w = int(img_h*stride_ratio), int(img_w*stride_ratio)
    # yy, xx = np.meshgrid(range(spl_h_stride, img_h-spl_h_stride, stride_h), range(spl_w_stride, img_w-spl_w_stride, stride_w))
    yy, xx = np.meshgrid(range(0, img_h, stride_h), range(0, img_w, stride_w))
    sample_mask[yy,xx] = 1

    random.seed(0)
    nsample = np.zeros(len(img_list), np.float32)
    for i, ((y, x), seg, img) in enumerate(img_list):
        nsample[i] = compute_grid_score(seg.copy(), threshold)  # to prevent samplable pixel amount smaller than it needs

    # sort grids by nsample and discard others
    sorted_idx = np.argsort(nsample)[::-1] # in descend order
    sorted_idx = sorted_idx[: min(tot_samples, sorted_idx.size)] # only keep the first tot_sample
    nsample = nsample[sorted_idx]
    keeped_img_list = [img_list[k] for k in list(sorted_idx)]
    maxidx = np.argmax(nsample, 0)
    if np.sum(nsample) == 0:
        nsample_per_img = np.zeros_like(nsample)
        print ('! Warning: not found any positive samples in this slide')
    else:
        nsample_per_img = nsample / np.sum(nsample) * tot_samples


    # with out assign all left to the first one we averagely assign to all together
    # Not sure if is a better option
    nsample_per_img = nsample_per_img.astype(np.int32)
    nsample_per_img[maxidx] += max(0, tot_samples - nsample_per_img.sum())
    
    print ('nsample_per_img', nsample_per_img) 
    samples = []
    for i, ((y, x), seg, img) in enumerate(keeped_img_list):
        seg[seg < threshold] = 0
        # mask seg
        tar_seg = seg * sample_mask
        sh, sw = np.nonzero(tar_seg)
        probs = tar_seg.copy().flatten()
        if probs.sum() != 0:
            probs = probs / probs.sum() # - 0.0000001  # take care of multinomial sampling
        else:
            continue
        # sampling per point
        idxs = np.random.choice(range(0, len(probs)), size=nsample_per_img[i], replace=True, p=probs)
        for idx in idxs:
            probs[idx] = 0 # set to zero so do not sample again
            h_id, w_id = np.unravel_index([idx], (seg.shape[0], seg.shape[1]))
            if (h_id >= img_h - spl_h_stride):
                h_id = img_h - spl_h_stride
            if (h_id < spl_h_stride):
                h_id = spl_h_stride
            if (w_id >= img_w - spl_w_stride):
                w_id = img_w - spl_w_stride
            if (w_id < spl_w_stride):
                w_id = spl_w_stride
            w_id = int(w_id)
            h_id = int(h_id)

            ## using sampled point as center point of the image patch.
            sample_img = img[(h_id-spl_h_stride) : (h_id+spl_h_stride), (w_id-spl_h_stride) : (w_id+spl_h_stride)].copy()
            sample_img_seg = seg[(h_id-spl_h_stride) : (h_id+spl_h_stride), (w_id-spl_h_stride) : (w_id+spl_h_stride)].copy()

            if not (sample_img.shape[0] == sample_size[0] and sample_img.shape[1] == sample_size[1]):
                print("Size of patch {} and image {}".format(sample_img.shape, img.shape))
                pdb.set_trace()

            # the left upper corner (act_y, act_x) of the sampled patch
            act_y = h_id + y
            act_x = w_id + x
            assert(sample_img.shape[0] == sample_size[0] and sample_img.shape[1] == sample_size[1])
            samples += [ ((act_x, act_y), sample_img, sample_img_seg) ]

    return samples

def gradient_merge(img1, img2, mask):
    y, x = np.where(mask != 0)
    y = np.unique(y)
    x = np.unique(x)
    res = img2.copy()
    if y.size > x.size:
        # overalpping is vertical
        for i, p in enumerate(x):
            w = i/(len(x)-1) 
            res[:,p] = res[:,p] * (1-w) + w * img1[:,p]
 
    else:
        # overlapping is horizontal
        for i, p in enumerate(y):
            w = i/(len(y)-1) 
            res[p,:] = res[p,:] * (1-w) + w * img2[p,:]
    
    return res

def visualize_sampling_points(target, locs, path=None):
    
    target_c = target.copy()
    for (x, y) in locs:
        target = cv2.circle(target, center=(x,y), radius=30, color=[0,255,100], thickness=4, lineType=8, shift=0)
    if path is not None:
        data = dict({'img': target_c, 'locs': locs})
        pickle.dump(data, open(path+'/visual_data.pickle','wb'))

    return target

def visualize_heatmap(wsi_seg, shape, stride, wsi_img, save_path):
    # import pdb; pdb.set_trace()
    # low_size = shape[0] // stride
    # res = np.zeros((low_size, low_size), dtype=np.float16)
    
    # for (x,y), im, seg in samples:
    #     v =  seg.sum() / (seg.shape[0] * seg.shape[1])
    #     x_, y_ = x // stride, y // stride
    #     res[y_, x_] = v
    # out = skimage.transform.pyramid_expand(res, upscale=stride, sigma=25)

    seg_img = misc.imresize(wsi_seg, 0.02)
    out = transform.pyramid_expand(seg_img, upscale=50, sigma=25)
    
    fig = plt.figure(figsize=(10, int(10*(out.shape[0]/out.shape[1]))))

    # plt.imshow(wsi_img)
    # plt.imshow(out, cmap=plt.cm.jet, alpha=0.5, interpolation='nearest')
    plt.imshow(out, cmap=plt.cm.jet)
    plt.axis('off')
    fig.tight_layout()
    fig.savefig(save_path+'_cam.jpg', bbox_inches='tight', pad_inches=0)
    contour_img = vis_overlay(wsi_img, wsi_seg.astype(np.float32)/wsi_seg.max(), threshold=0.2)
    misc.imsave(save_path + '_contour.jpg', contour_img)
    
    return out


def vis_overlay(im, pred, threshold=-1):
    res = []
    pred = pred > threshold
    im = im.astype(np.uint8)
    
    _, contours, hierarchy = cv2.findContours(pred.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(im, contours, -1, (0,255,0), 10) 

    return im

# if __name__ == "__main__":
    # # data = pickle.load(open('/home/zizhaozhang/work/mdnet_checkpoints_remote/segmentation/test_slides/segmentor_drop0.0_ignorew0.5/116652/visual_data.pickle', 'rb'))
    # import matplotlib.cm as cm
    # from skimage import transform
    # # res = visualize_sampling_points(data['img'], data['locs'])
    # # res = misc.imresize(res, 0.5)
    # # plt.ion()
    # # plt.imshow(res)

    # root = '/home/zizhaozhang/work/mdnet_checkpoints_remote/segmentation/test_slides/segmentor_drop0.0_ignorew0.5/116652_overlap0/'

    # wsi_img = misc.imread(root+'116652_thumb.jpg')
    # # alpha_img = misc.imread(root+'116652_heatmap.jpg')
    # wsi_seg = misc.imread(root+'116652_seg.jpg')
    # seg_img = misc.imresize(wsi_seg, 0.02)
    # out = transform.pyramid_expand(seg_img, upscale=50, sigma=25)
    # # wsi_img = misc.imresize(wsi_img, alpha_img.shape)
    # fig = plt.figure(figsize=(20,20))

    # plt.imshow(wsi_img)
    # plt.imshow(out, cmap=plt.cm.Greys_r, alpha=0.9, interpolation='nearest')
    # # plt.imshow(out, alpha=0.5)
    # # plt.set_cmap(cm.jet)#cm.Greys_r)
    # plt.axis('off')
    # fig.tight_layout()
    # plt.show()