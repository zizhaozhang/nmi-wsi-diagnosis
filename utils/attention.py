from __future__ import print_function


import numpy as np
import skimage
import skimage.transform
import skimage.io
import os, sys, traceback, pdb
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import deepdish, shutil
import scipy.misc as misc

def normalize(x, thre=0.3):
    x[x < thre * np.max(x[:])] = 0
    x = x / (x.max() + 0.1)
    # x = x / (x.sum())
    # x[x< thre] = 0
    return x
    
def generate_attention_sequence_single(images, seq_atts, 
                    cam_atts, pred_sents, savedir, upscale=0, sigma=25 ):

    # different from generate_attention_sequence(), this function save each attention maps as the single image

    print ('--> visualizing attention maps at '+savedir+'_singleattvis')
    num_data = len(images)
    print ('--> saving attention maps for '),
    for ii in range(num_data):
        try:
            img = images[ii]
            att  = seq_atts[ii] # sequence attention
            cam_att = cam_atts[ii]   # class-specific attention
            words = pred_sents[ii]['caption'].split()
            name = pred_sents[ii]['image_id']
            spat_width = int(np.sqrt(cam_att.size))
            if upscale == 0:
                upscale = int(img.shape[0] / spat_width)
            # upscale = 16 #TODO
            print (name+',', end='')

            if not os.path.isdir(savedir+'_singleattvis'): os.mkdir(savedir+'_singleattvis')
            save_img_path = os.path.join(savedir+'_singleattvis', name)
            
            # assert len(words) == len(att), 'size not equal'
            if os.path.isfile(save_img_path):
                shutil.rmtree(save_img_path)
            os.mkdir(save_img_path)
            
            # lstm visual attention
            for ii in range(len(words)):
                if words[ii] in [':', '.']: 
                    continue 
                
                lab = words[ii]
                this_alpha = att[ii,...].copy()
                tot_weight = '%.3f'.format(this_alpha.sum())
                this_alpha = normalize(this_alpha)

                fig = plt.figure(figsize=(20,20))
                plt.imshow(img)
                alpha_img = skimage.transform.pyramid_expand(this_alpha.reshape(spat_width,spat_width), upscale=upscale, sigma=sigma) # TODO
                alpha_img = skimage.transform.resize(alpha_img, [img.shape[0], img.shape[1]])
                #print alpha_img.shape
                plt.imshow(alpha_img, alpha=0.8)
                plt.set_cmap(cm.Greys_r)
                
                plt.axis('off')
                fig.tight_layout()
                fig.savefig('{}/{}_att_{}_{}.png'.format(save_img_path, name, lab, tot_weight))
            
            # class-specific attention
            fig = plt.figure(figsize=(20,20))
            plt.imshow(img)
            plt.axis('off')
            cam_att = normalize(cam_att)
            cam_att[cam_att>1.0] = 1.0
            cam_att = cam_att.reshape(spat_width, spat_width)
            cam_att = skimage.transform.pyramid_expand(cam_att.reshape(spat_width, spat_width), upscale=upscale, sigma=sigma)
            cam_att = skimage.transform.resize(cam_att, [img.shape[0], img.shape[1]])
            plt.imshow(cam_att, cmap=plt.cm.jet, alpha=0.5, interpolation='nearest')
            fig.tight_layout()
            # pdb.set_trace()
            fig.savefig(save_img_path+'_cam.png')

        except Exception as e:
                print ('--> fail to generate attention maps ... ')
                print (e)
                traceback.print_exc(file=sys.stdout)
                # print (words)
                # print (len(words), len(att))

def generate_attention_sequence(images, seq_atts, 
                    cam_atts, pred_sents, savedir, upscale=0, sigma=25 ):

    left  = 0.05  # the left side of the subplots of the figure
    right = 0.95    # the right side of the subplots of the figure
    bottom = 0.05   # the bottom of the subplots of the figure
    top = 0.95      # the top of the subplots of the figure
    wspace = 0.05   # the amount of width reserved for blank space between subplots
    hspace = 0.05   # the amount of height reserved for white space between subplots

    # savedir = savedir + '_nonormalize' #TODO
    print ('--> visualizing attention maps at '+savedir+'_attvis', flush=True)
    num_data = len(images)
    for ii in range(num_data):
        try:
            img = images[ii]
            att  = seq_atts[ii] # sequence attention
            cam_att = cam_atts[ii]   # class-specific attention
            words = pred_sents[ii]['caption'].split()
            name = pred_sents[ii]['image_id']
            spat_width = int(np.sqrt(cam_att.size))
            if upscale == 0:
                upscale = int(img.shape[0] / spat_width)
            # upscale = 16 #TODO

            if not os.path.isdir(savedir+'_attvis'): os.mkdir(savedir+'_attvis')
            save_img_path = os.path.join(savedir+'_attvis', name)
            
            # assert len(words) == len(att), 'size not equal'
            n_words = att.shape[0]
            if n_words != len(words):
                print ('attention and word {}!={}, skip'.format(n_words, len(words)))
                continue

            w = np.round(np.sqrt(n_words))
            h = np.ceil(np.float32(n_words) / w)

            fig = plt.figure(figsize=(20,20))
            smooth = True

            plt.subplot(w, h, 1)
            plt.imshow(img)
            plt.axis('off')
            # lstm visual attention
            pp = 0
            for ii in range(len(words)):
                if words[ii] in [':', '.']: 
                    continue 
                ax1= plt.subplot(w, h, pp+1)
                pp += 1
                lab = words[ii] 

                plt.imshow(img)
                this_alpha = att[ii,...].copy()
                att_weight = "%.3f"%(this_alpha.sum())
                # remove low value, do it before removing borders, otherwise the error will be magnified after normalization
                
                this_alpha = normalize(this_alpha)

                # a little post processing here, remove all attention around the border
                # this_alpha[0,:] = 0
                # this_alpha[-1,:] = 0
                # this_alpha[:,0] = 0
                # this_alpha[:,-1] = 0
                plt.text(0, 20, lab, backgroundcolor='white', color='black',  fontsize=20)
                if smooth:
                    alpha_img = skimage.transform.pyramid_expand(this_alpha.reshape(spat_width,spat_width), upscale=upscale, sigma=sigma) # TODO
                    alpha_img = skimage.transform.resize(alpha_img, [img.shape[0], img.shape[1]])
                else:
                    alpha_img = skimage.transform.resize(this_alpha.reshape(spat_width,spat_width), [img.shape[0], img.shape[1]])
                #print alpha_img.shape
                plt.imshow(alpha_img, alpha=0.8)
                plt.set_cmap(cm.Greys_r)
                
                plt.axis('off')
                plt.subplots_adjust(wspace=wspace, hspace=hspace,left=left, bottom=bottom, right=right, top=top)

                fig.tight_layout()
                #plt.show()
                fig.savefig(save_img_path+'_att.png')
            
            # class-specific attention
            fig = plt.figure(figsize=(20,20))
            plt.imshow(img)
            plt.axis('off')
            cam_att = normalize(cam_att)
            cam_att[cam_att>1.0] = 1.0
            cam_att = cam_att.reshape(spat_width, spat_width)
            cam_att = skimage.transform.pyramid_expand(cam_att.reshape(spat_width, spat_width), upscale=upscale, sigma=sigma)
            cam_att = skimage.transform.resize(cam_att, [img.shape[0], img.shape[1]])
            plt.imshow(cam_att, cmap=plt.cm.jet, alpha=0.5, interpolation='nearest')
            fig.tight_layout()
            # pdb.set_trace()
            fig.savefig(save_img_path+'_cam.png')

        except Exception as e:
                print ('--> fail to generate attention maps ... ')
                print (e)
                traceback.print_exc(file=sys.stdout)
                # print (words)
                # print (len(words), len(att))

def generate_attention_sequence2(images, seq_atts, 
        cam_atts, pred_sents, savedir, upscale=0, sigma=25, threshold=0, upsample_mode='gaussian'):
    ## this is for topic mdnet with attention on the concepts
    ## Drap soft attention maps
    
    max_len = 6 # euqal to num_feature
    left  = 0.01  # the left side of the subplots of the figure
    right = 0.99    # the right side of the subplots of the figure
    bottom = 0.01   # the bottom of the subplots of the figure
    top = 0.99      # the top of the subplots of the figure
    wspace = 0.05   # the amount of width reserved for blank space between subplots
    hspace = 0.05 # the amount of height reserved for white space between subplots
    if not os.path.isdir(savedir+'_attvis'): 
        os.mkdir(savedir+'_attvis')
 
    for ii in range(len(images)):
        try:
            img = images[ii]
            att = seq_atts[ii] # sequence attention
            cam_att = cam_atts[ii]   # class-specific attention
            name = pred_sents[ii]['image_id']
            sent = pred_sents[ii]['caption']
            spat_width = int(np.sqrt(cam_att.size))
            if upscale == 0:
                upscale = int(img.shape[0] / spat_width)
 
            save_img_path = os.path.join(savedir+'_attvis', name)
            # if os.path.isfile(save_img_path+'_att.png'):
            #     continue
            # if 'H12' not in name:
            #     continue
     
            print ('processing {}/{} {}'.format(ii, len(images), name), flush=True)
            # concept attention
            max_len = att.shape[0]
            fig, ax = plt.subplots(1, max_len+1, sharex='col',sharey='row',figsize=(10*(max_len+1),10))
             
            ax[0].imshow(img)
            ax[0].axis('off')
            # draw every attention 
            for i in range(max_len):
                # ax1= plt.subplot(1, max_len+1, i+2)
                p = ax[i+1]
                p.imshow(img)
                this_alpha = att[i,...].copy()
                att_weight = "%.3f"%(this_alpha.sum())
                # may not need this normalization            
                # this_alpha = normalize(this_alpha)
                this_alpha = (this_alpha - this_alpha.min()) / (this_alpha.max() - this_alpha.min())
                this_alpha[this_alpha < threshold] = 0
                 
                if upsample_mode == 'gaussian':
                    alpha_img = skimage.transform.pyramid_expand(this_alpha.reshape(spat_width, spat_width), upscale=upscale, sigma=sigma)
                    alpha_img = skimage.transform.resize(alpha_img, [img.shape[0], img.shape[1]])
                elif upsample_mode == 'nearest':
                    alpha_img = this_alpha.reshape(spat_width, spat_width)
                    alpha_img = misc.imresize(alpha_img, [img.shape[0], img.shape[1]], interp='nearest')
 
                # print alpha_img.shape
                plt.set_cmap(cm.Greys_r) 
                p.imshow(alpha_img, alpha=0.8)
                p.axis('off')
            fig.subplots_adjust(wspace=wspace, hspace=hspace, left=left, bottom=bottom+0.1, right=right, top=top)
            # fig.tight_layout()
            fig.text(0, 0, sent, backgroundcolor='white', color='black', fontsize=40)
            fig.savefig(save_img_path+'_att.png')
            plt.close(fig)
 
            # # class-specific attention
            # fig = plt.figure(figsize=(20,20))
            # plt.imshow(img)
            # plt.axis('off')
 
            # cam_att[cam_att < threshold] = 0
            # cam_att = (cam_att - cam_att.min()) / (cam_att.max() - cam_att.min())
            # cam_att = cam_att.reshape(spat_width, spat_width)
            # cam_att = skimage.transform.pyramid_expand(cam_att.reshape(spat_width, spat_width), upscale=upscale, sigma=sigma)
            # cam_att = skimage.transform.resize(cam_att, [img.shape[0], img.shape[1]])
            # plt.imshow(cam_att, cmap=cm.Greys_r, alpha=0.8)
            # # plt.imshow(cam_att, cmap=plt.cm.jet, alpha=0.5, interpolation='nearest')
            # fig.tight_layout()
            # # pdb.set_trace()
            # fig.savefig(save_img_path+'_cam.png')
            # plt.close(fig)
 
        except Exception as e:
            print ('--> fail to generate attention maps ... ')
            print (e)
            traceback.print_exc(file=sys.stdout)

def generate_attention_sequence3(images, seq_atts, 
        cam_atts, pred_sents, savedir, upscale=0, sigma=25, threshold=0, upsample_mode='gaussian'):
    ## this is for topic mdnet with attention on the concepts
    # draw binary mask and light off irrelvenat regions

    max_len = 6 # euqal to num_feature
    left  = 0.01  # the left side of the subplots of the figure
    right = 0.99    # the right side of the subplots of the figure
    bottom = 0.01   # the bottom of the subplots of the figure
    top = 0.99      # the top of the subplots of the figure
    wspace = 0.05   # the amount of width reserved for blank space between subplots
    hspace = 0.05 # the amount of height reserved for white space between subplots

    if not os.path.isdir(savedir+'_attvis'): 
        os.mkdir(savedir+'_attvis')

    for ii in range(len(images)):
        try:
            img = images[ii]
            att = seq_atts[ii] # sequence attention
            cam_att = cam_atts[ii]   # class-specific attention
            name = pred_sents[ii]['image_id']
            sent = pred_sents[ii]['caption']
            spat_width = int(np.sqrt(cam_att.size))
            if upscale == 0:
                upscale = int(img.shape[0] / spat_width)

            save_img_path = os.path.join(savedir+'_attvis', name)
            # if os.path.isfile(save_img_path+'_att.png'):
            #     continue
            if 'H12' not in name:
                continue
    
            print ('processing {}/{} {}'.format(ii, len(images), name), flush=True)
            # concept attention
            max_len = att.shape[0]
            fig, ax = plt.subplots(1, max_len+1, sharex='col',sharey='row',figsize=(10*(max_len+1),10))
            
            ax[0].imshow(img)
            ax[0].axis('off')
            # draw every attention 
            for i in range(max_len):
                # ax1= plt.subplot(1, max_len+1, i+2)
                p = ax[i+1]
                # p.imshow(img)
                this_alpha = att[i,...].copy()
                att_weight = "%.3f"%(this_alpha.sum())
                # may not need this normalization            
                # this_alpha = normalize(this_alpha)
                this_alpha = (this_alpha - this_alpha.min()) / (this_alpha.max() - this_alpha.min())
                this_alpha[this_alpha < threshold] = 0
                
                if upsample_mode == 'gaussian':
                    alpha_img = skimage.transform.pyramid_expand(this_alpha.reshape(spat_width, spat_width), upscale=upscale, sigma=sigma)
                    alpha_img = skimage.transform.resize(alpha_img, [img.shape[0], img.shape[1]])
                elif upsample_mode == 'nearest':
                    alpha_img = this_alpha.reshape(spat_width, spat_width)
                    alpha_img = misc.imresize(alpha_img, [img.shape[0], img.shape[1]], interp='nearest')
                
                alpha_img[alpha_img >= 0.5] = 1
                alpha_img[alpha_img < 0.5] = 0
                alpha_img = np.tile(alpha_img[:, :, np.newaxis], (1,1,3))
                alpha = 0.2
                alpha_img = (alpha_img * (1 - alpha) + alpha) * (img.astype(np.float32) / 255)
                # print alpha_img.shape
                plt.set_cmap(cm.Greys_r) 
                p.imshow(alpha_img)
                p.axis('off')
            fig.subplots_adjust(wspace=wspace, hspace=hspace, left=left, bottom=bottom+0.1, right=right, top=top)
            # fig.tight_layout()
            fig.text(0, 0, sent, backgroundcolor='white', color='black', fontsize=40)
            fig.savefig(save_img_path+'_att.png')
            plt.close(fig)

        except Exception as e:
            print ('--> fail to generate attention maps ... ')
            print (e)
            traceback.print_exc(file=sys.stdout)
