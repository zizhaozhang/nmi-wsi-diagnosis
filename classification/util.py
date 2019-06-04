import os, sys, pdb
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
import cv2, shutil, json
from sklearn.metrics import confusion_matrix

class VIS:
    def __init__(self, save_path):

        self.path = save_path
        # TODO
        self.preds = []
        self.gts = []
        self.score_history = {}
        # load history
        if os.path.isfile(os.path.join(self.path, 'accuracy.json')):
            with open(os.path.join(self.path, 'accuracy.json'), 'r') as f:
                history = json.load(f)
                for k, v in history.items():
                    self.score_history[k] = v
        self.avg = 0
        self.tot = 0
    def reset(self):
        self.preds = []
        self.gts = []
        self.avg = 0
        self.tot = 0

    def add_sample(self, pred, gt):
        score_mean = compute_accuracy(pred, gt)
        self.avg = self.avg * self.tot + (score_mean * pred.shape[0])
        self.tot += pred.shape[0]
        self.avg /= self.tot
        self.preds.append(pred)
        self.gts.append(gt)
        return self.avg

    def compute_scores(self, suffix=0):

        preds = np.concatenate(self.preds, axis=0)
        gts = np.concatenate(self.gts, axis=0)
        score = compute_accuracy(preds, gts) * 100.0
        cm = confusion_matrix(gts, preds)

        print ('-'*20)
        print ('overall accuracy: {} '.format(score))
        print ('confusion_matrix: \n {}'.format(cm))
        print ('-'*20)

        data = {'accuracy': '%.3f' % (score), 'conf_matrix': cm.tolist()}
        self.score_history['%.10d' % suffix] = data
        json.dump(self.score_history, open(os.path.join(self.path, 'accuracy.json'),'w'), indent=2, sort_keys=True)

        return score

def compute_accuracy(y_pred, y_true):
    return np.mean(np.equal(y_pred, y_true))


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
