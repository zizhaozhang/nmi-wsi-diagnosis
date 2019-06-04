import os, sys, pdb
from skimage import io, transform
import numpy as np

import cv2, shutil, json
import tensorflow as tf
from matplotlib import pyplot as plt
import seaborn as sns

class VIS:
    def __init__(self, save_path):

        self.path = save_path
        # TODO
        self.semantic_label = None
        self.mean_iu = []
        self.cls_iu = []
        self.score_history = {}
        # load history
        if os.path.isfile(os.path.join(self.path, 'meanIU.json')):
            with open(os.path.join(self.path, 'meanIU.json'), 'r') as f:
                history = json.load(f)
                for k, v in history.items():
                    self.score_history[k] = v

    # def save_seg(self, label_im, name, im=None, gt=None):

    #     seg = Image.fromarray(label_im.astype(np.uint8), mode='P') # must convert to int8 first
    #     seg.palette = copy.copy(self.palette)
    #     if gt is not None or im is not None:
    #         gt = Image.fromarray(gt.astype(np.uint8), mode='P') # must convert to int8 first]
    #         gt.palette = copy.copy(self.palette)
    #         im = Image.fromarray(im.astype(np.uint8), mode='RGB')
    #         I = Image.new('RGB', (label_im.shape[1]*3, label_im.shape[0]))
    #         I.paste(im,(0,0))
    #         I.paste(gt,(256,0))
    #         I.paste(seg,(512,0))
    #         I.save(os.path.join(self.path, name))
    #     else:
    #         seg.save(os.path.join(self.path, name))
    def reset(self):
        self.mean_iu = []
        self.cls_iu = []

    def add_sample(self, pred, gt):
        score_mean, score_cls = mean_IU(pred, gt)

        self.mean_iu.append(score_mean)
        if len(score_cls) != 2:
            score_cls.append(1)
        self.cls_iu.append(score_cls)

        return score_mean

    def compute_scores(self, suffix=0):
        meanIU = np.mean(np.array(self.mean_iu))
        meanIU_per_cls = np.mean(np.asarray(self.cls_iu), axis=0)
        print ('-'*20)
        print ('overall mean IU: {} '.format(meanIU))
        print ('mean IU per class')
        for i, c in enumerate(meanIU_per_cls):
            print ('\t class {}: {}'.format(i,c))
        print ('-'*20)

        data = {'mean_IU': '%.2f' % (meanIU), 'mean_IU_cls': ['%.2f'%(a) for a in meanIU_per_cls.tolist()]}
        self.score_history['%.10d' % suffix] = data
        json.dump(self.score_history, open(os.path.join(self.path, 'meanIU.json'),'w'), indent=2, sort_keys=True)



class VISRecall:
    def __init__(self, save_path):

        self.path = save_path
        # TODO
        self.semantic_label = None
        self.mean_recall_pos = []
        self.mean_recall_neg = []
        self.cls_iu = []
        self.score_history = {}
        # load history
        if os.path.isfile(os.path.join(self.path, 'Recall.json')):
            with open(os.path.join(self.path, 'Recall.json'), 'r') as f:
                history = json.load(f)
                for k, v in history.items():
                    self.score_history[k] = v

    def reset(self):
        self.mean_iu = []
        self.cls_iu = []

    def add_sample(self, pred, gt, mask):
        score, is_positive = Recall(pred, gt, mask)
        if is_positive:
            self.mean_recall_pos.append(score)
        else:
            self.mean_recall_neg.append(score)

        return np.mean(score)

    def compute_scores(self, suffix=0):
        mean_recall_pos = np.mean(np.asarray(self.mean_recall_pos), axis=0) #[sample, thresholds]
        mean_recall_neg = np.mean(np.asarray(self.mean_recall_neg), axis=0)

        data = {
            'recall_pos': ['%.3f'%(a) for a in mean_recall_pos.tolist()],
            'recall_neg': ['%.3f'%(a) for a in mean_recall_neg.tolist()][::-1],
            }

        self.score_history['%.10d' % suffix] = data
        json.dump(self.score_history, open(os.path.join(self.path, 'recall_curves.json'),'w'), indent=2, sort_keys=True)

        # plt.figure()
        # sns.set_style("white")
        # x = [thre / 100 for thre in range(10,90)]
        # # import pdb; pdb.set_trace()
        # plt.step(x, data['recall_pos'],
        #         color='navy',linewidth=2,
        #         label='Tumor')

        # plt.step(x, data['recall_neg'],
        #         color='green',linewidth=2,
        #         label='Non-Tumor')

        # # plt.xlim([0.0, max(rec)])
        # # plt.ylim([0.0, max(prec)+0.05])
        # plt.xlabel('Recall',fontsize=6)
        # plt.ylabel('Thresholds',fontsize=6)
        # plt.title('Tumor Detection', fontsize=6)
        # plt.legend(loc="lower left", fontsize=6)

        # plt.savefig(os.path.join(self.path, 'recall_curves.pdf'), bbox_inches='tight')


def Recall(pred, gt, mask):

    maxv = gt[:].max()
    cv = 1 if maxv == 1 else 0 # it is a positive class or negative
    if cv == 0:
        # if negative class
        gt = mask
        pred = 1 - pred

    tot = np.sum(gt == 1)
    recalls = []
    for thre in range(10,90):
        thre /= 100
        v = np.sum((gt == 1) * ((pred > thre).astype(np.int8)  == 1))
        assert(v / tot <= 1)
        recalls.append(v / tot)

    return recalls, cv == 1


def vis_overlay(imgs, labels, preds, use_mask=True, threshold=None):
    def compute_alpha_img(img, label, color='r'):
        if color == 'r':
            tmp = [label[:,:,np.newaxis], np.zeros((h,w,1),np.int64), np.zeros((h,w,1),np.int64)]
        elif color == 'b':
            tmp = [label[:,:,np.newaxis],  np.zeros((h,w,1),np.int64), 1-label[:,:,np.newaxis]]

        return np.concatenate(tmp, 2) * 255.0
        # idx = np.concatenate(tmp, 2) #* 255.0
        # alp = im.copy()
        # alp[idx > 0] = 255
        # return alp

    res = []
    alpha = 0.7
    for im, lab, pre in zip(imgs, labels, preds):
        lab = lab > 0
        im = ((im - im.min()) / (im.max() - im.min()) * 255.0 ).astype(np.uint8)

        _, contours, hierarchy = cv2.findContours(lab.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(im, contours, -1, (0,255,0), 3)
        h,w = pre.shape
        # alp = compute_alpha_img(im, lab, color='b')
        # im = cv2.addWeighted(im, alpha, alp.astype(np.uint8),1-alpha, 0)

        if threshold is not None:
            # -1: visualize probablity maps
            pre = pre > threshold
        if use_mask:
            # conver to rgb
            alp = compute_alpha_img(im, pre, color='b')
            im = cv2.addWeighted(im,alpha,alp.astype(np.uint8),1-alpha,0)
        else:
            pre = pre > 0
            _, contours, hierarchy = cv2.findContours(pre.astype(np.uint8),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(im, contours, -1, (255,0,0), 5)

        res.append(im)

    return res

def make_grid(array_list, ncols=3):
    array = np.array(array_list)
    nindex, height, width, intensity = array.shape
    nrows = nindex//ncols
    assert nindex == nrows*ncols
    # want result.shape = (height*nrows, width*ncols, intensity)
    result = (array.reshape(nrows, ncols, height, width, intensity)
              .swapaxes(1,2)
              .reshape(height*nrows, width*ncols, intensity))
    return result

def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = tf.flatten(y_true)
    y_pred_f = tf.flatten(y_pred)
    intersection = tf.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.sum(y_true_f) + tf.sum(y_pred_f) + smooth)

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def save_model(sess, saver, checkpoint_dir, model_name, step):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    saver.save(sess, os.path.join(checkpoint_dir, model_name), global_step=step)

def load_model(sess, saver, checkpoint_dir):
    print("[*] Reading checkpoints...")
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
        return True
    return False

def gen_thumbnail(img, thumb_max=1000):
    img_shape = img.shape[:2]
    max_side = max(img_shape)

    if max_side <= thumb_max:
        return img

    resize_ratio = thumb_max * 1.0 / max_side
    new_shape = [int(x*resize_ratio) for x in img_shape]

    return transform.resize(img, new_shape)

def add_pad16(img):
    input_rows = img.shape[0]
    input_cols = img.shape[1]
    assert(input_rows >= 16 and input_cols >= 16)

    pad_top, pad_bottom, pad_left, pad_right = 0, 0, 0, 0
    row_need, col_need = 0, 0
    if input_rows % 16 != 0:
        row_need = 16 - input_rows % 16
        assert (row_need > 0)
        if row_need % 2 != 0:
            pad_top, pad_bottom = int(row_need/2), int(row_need/2) + 1
        else:
            pad_top, pad_bottom = int(row_need/2), int(row_need/2)
    row_pad = (pad_top, pad_bottom)

    if input_cols % 16 != 0:
        col_need = 16 - input_cols % 16
        assert (col_need > 0)
        if col_need % 2 != 0:
            pad_left, pad_right = int(col_need/2), int(col_need/2) + 1
        else:
            pad_left, pad_right = int(col_need/2), int(col_need/2)
    col_pad = (pad_left, pad_right)

    padded_img = np.zeros((input_rows+row_need, input_cols+col_need, img.shape[2]), dtype=img.dtype)
    for channel in range(img.shape[2]):
        padded_img[:,:,channel] = np.lib.pad(img[:,:,channel], (row_pad, col_pad), 'reflect')

    return padded_img, row_pad, col_pad


def remove_pad(padded_img, row_pad, col_pad):
    pad_top, pad_bottom = row_pad
    pad_left, pad_right = col_pad

    row_start = pad_top
    row_end = padded_img.shape[0] - pad_bottom

    col_start = pad_left
    col_end = padded_img.shape[1] - pad_right
    return padded_img[row_start:row_end, col_start:col_end]

def display_prediction(img, mask, pred):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    a=fig.add_subplot(1,3,1)
    imgplot = plt.imshow(img)
    a.set_title('Img')
    b=fig.add_subplot(1,3,2)
    imgplot = plt.imshow(mask, cmap='gray')
    b.set_title('Mask')
    c=fig.add_subplot(1,3,3)
    imgplot = plt.imshow(pred, cmap='gray')
    c.set_title('Predict')
    plt.show()

def mean_IU(eval_segm, gt_segm):
    '''
    (1/n_cl) * sum_i(n_ii / (t_i + sum_j(n_ji) - n_ii))
    '''

    check_size(eval_segm, gt_segm)

    cl, n_cl   = union_classes(eval_segm, gt_segm)
    _, n_cl_gt = extract_classes(gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

    IU = list([0]) * n_cl

    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]

        if (np.sum(curr_eval_mask) == 0) or (np.sum(curr_gt_mask) == 0):
            continue

        n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        t_i  = np.sum(curr_gt_mask)
        n_ij = np.sum(curr_eval_mask)

        IU[i] = n_ii / (t_i + n_ij - n_ii)

    mean_IU_ = np.sum(IU) / n_cl_gt
    return mean_IU_, IU
def extract_both_masks(eval_segm, gt_segm, cl, n_cl):
    eval_mask = extract_masks(eval_segm, cl, n_cl)
    gt_mask   = extract_masks(gt_segm, cl, n_cl)

    return eval_mask, gt_mask

def extract_classes(segm):
    cl = np.unique(segm)
    n_cl = len(cl)

    return cl, n_cl

def union_classes(eval_segm, gt_segm):
    eval_cl, _ = extract_classes(eval_segm)
    gt_cl, _   = extract_classes(gt_segm)

    cl = np.union1d(eval_cl, gt_cl)
    n_cl = len(cl)

    return cl, n_cl

def extract_masks(segm, cl, n_cl):
    h, w  = segm_size(segm)
    masks = np.zeros((n_cl, h, w))

    for i, c in enumerate(cl):
        masks[i, :, :] = segm == c

    return masks

def segm_size(segm):
    try:
        height = segm.shape[0]
        width  = segm.shape[1]
    except IndexError:
        raise

    return height, width

def check_size(eval_segm, gt_segm):
    h_e, w_e = segm_size(eval_segm)
    h_g, w_g = segm_size(gt_segm)

    if (h_e != h_g) or (w_e != w_g):
        raise ValueError('Uneuqal image and mask size')
