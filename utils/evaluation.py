"""
 * @author [author]
 * @email [example@mail.com]
 * @date 2017-03-06 10:51:17
 * @desc [description]
"""
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import numpy as np
# from termcolor import colored
import json
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.3f')
import sys, os, copy, pdb
import scipy.io as io
import skimage
import scipy
import deepdish
from metric.pycocotools.coco2 import COCO # coco2 is my revised version for MDNet  
from metric.pycocoevalcap.eval import COCOEvalCap
import metric.preprocessing as preprocessing
import h5py
from .attention import generate_attention_sequence, generate_attention_sequence_single, generate_attention_sequence2
import pickle


class Evaluation:
    def __init__(self, opt, data_loader):
        self.data_loader = data_loader
        self.cnn_pred_list = []
        self.cnn_gt_list = []
        self.lstm_acc = 0
        self.lstm_pred = []
        self.best_acc = 0
        self.att_prob_maps = []
        self.cam_prob_maps = []
        self.images = []
        
        self.verbose_att = opt.test_mode


    def add_accuracy(self, txt_preds, cls_preds, txt_labels, 
            cls_labels, name_list, gt_sents,
            image, att_prob, cam_prob, 
            verbose=False):
        """ add a batch accuracy
            Input: 
                txt_preds [batch_size, max_subseq_len]
                cls_preds [batch_size]
                att_prob [batch_size, max_subseq_len, conv_feat_spat] or [batch_size, num_feature, conv_feat_spat]
                cam_prob [batch_size, conv_feat_spat ]
                txt_stop [batch_size, num_feature] in range[0,2], stop if equal to 2
            Output:

            Note that Python is row-priority. When spliting batch_size*num_feature, we need to reshape. See following...
        """


        def _convert_to_indices(timestep_len):
            n = len(timestep_len)
            max_len = self.data_loader.max_subseq_len
            out = []
            for m in range(len(timestep_len)):
                out += [s+m*max_len for s in range(timestep_len[m])]
            return out
            
        ## classification outputs
        if cls_preds is not None and cls_labels is not None:
            cls_pred_labs = np.argmax(cls_preds, 1)
            self.cnn_pred_list += list(cls_pred_labs)
            self.cnn_gt_list += list(cls_labels)

        ## text outputs
        if txt_preds.shape[0] != cls_preds.shape[0]:
            txt_preds = np.reshape(txt_preds, newshape=[self.data_loader.num_feature, -1, txt_preds.shape[1]])
            txt_preds = np.transpose(txt_preds, (1,0,2))
        # seq = self.data_loader.decode_seq(txt_preds)
        pred_sents, timestep_len = self.data_loader.convert_to_text_list(txt_preds)

        local_lstm_pred = []
        for i, item in enumerate(pred_sents):
            par = {
                    'image_id': name_list[i],
                    'caption': pred_sents[i],
                    'step_len': timestep_len[i]
                }
            
            if verbose and cls_pred_labs[i] == cls_labels[i]:  
                print ('-'*20 + ' ' + name_list[i])
                print ('CAPTION: ' + pred_sents[i])
                if gt_sents != None: 
                    print('GROUNDTH: ' + gt_sents[i])
                    par['groundtruth'] = gt_sents[i]
            local_lstm_pred.append(par)

        self.lstm_pred += local_lstm_pred
    
        if len(local_lstm_pred) != image.shape[0]:
            import pdb; pdb.set_trace()

        if self.verbose_att:
            if att_prob.shape[1] > self.data_loader.num_feature:
                ''''it is the MDNet '''
                ## extraxt valid attention maps based on timestep_len
                att_prob = np.reshape(att_prob.copy(), newshape=[self.data_loader.num_feature, -1, att_prob.shape[1], att_prob.shape[2]])
                att_prob = np.transpose(att_prob, (1,0,2,3))
                att_prob = np.reshape(att_prob, newshape=[att_prob.shape[0], att_prob.shape[1]*att_prob.shape[2], att_prob.shape[3] ])
                assert(len(timestep_len) == att_prob.shape[0])
                assert(len(timestep_len[0]) == self.data_loader.num_feature)
                 # for each data in the batch
                for i in range(len(timestep_len)):
                    indices = _convert_to_indices(timestep_len[i])
                    tmp = att_prob[i,indices,:]

                    self.att_prob_maps += [tmp]
                    self.cam_prob_maps += [cam_prob[i]]
                    self.images += [image[i].astype(np.uint8)]
            else:
                ''''it is the TopicMDNet '''
                assert(hasattr(self.data_loader, 'txt_stops'))
                # att_prob [batch_size, num_feature, conv_feat_spat]
                for i in range(att_prob.shape[0]):
                    # at least to be one
                    length = np.sum(self.data_loader.txt_stops[i] == self.data_loader.stop_label-1) + 1
                    self.att_prob_maps += [att_prob[i,:length,:]]
                    self.cam_prob_maps += [cam_prob[i]]
                    self.images += [image[i].astype(np.uint8)]
                    
        
    def summary_overall_evaluation(self, save_path, 
                                    no_eval=False, to_demo_input=False, 
                                    save_to_mat=False, single_vis=False,
                                    test_groundtruth_json='./metric/test_annotation_striped.json'):
        
        print ('--> save all outputs at', save_path)
        best_surfix = ''
        if self.cnn_gt_list != [] and self.cnn_pred_list != []:
        
            print ('='*10 + ' cnn metric ' + '='*10)
            conf  = confusion_matrix(self.cnn_gt_list, self.cnn_pred_list)
            print ('Confusion matrix: ')
            print (conf)
            acc = accuracy_score(self.cnn_gt_list, self.cnn_pred_list)
            
            if acc >= self.best_acc: 
                self.best_acc = acc
                best_surfix = '_BEST%.2f' % (acc)
            print ('Overall accuracy: ' + str(acc) )
            print ('='*10 + ' lstm evaluation metric ' + '='*10)

        resFile = save_path+'_captions'    
        with open(resFile+'.json','w') as f:
            json.dump(self.lstm_pred,f, indent=4, sort_keys=True)

        if not no_eval:
            coco = COCO(test_groundtruth_json)
            cocoRes = coco.loadRes(resFile+'.json')
            cocoEval = COCOEvalCap(coco, cocoRes)
            cocoEval.params['image_id'] = cocoRes.getImgIds()
            cocoEval.evaluate()

            out = {}
            for metric, score in cocoEval.eval.items():
                out[metric] = score
            json.dump(out, open(resFile+'_metrics{}.json'.format(best_surfix), 'w'), indent=4, sort_keys=True)
            # print out
            print ('-'*50)
            for item in out.keys():
                if 'Accuracy' in item:
                    print ("LSTM Accuracy: \t %s" % (str(out[item][2])))
                else:
                    print ("%s: \t %s" % (item, str(out[item])))
            print ('-'*50)

        if self.verbose_att:
            assert (len(self.lstm_pred) == len(self.images))
            print ('--> processing attention maps')
            if to_demo_input: self.to_demo_input(save_path)
            
            if save_to_mat: self.save_to_mat(save_path)

            with open(save_path+'_attentions_train.pickle','wb') as f:
                pickle.dump({'images': self.images, 
                                'att_prob_maps':self.att_prob_maps, 
                                'cam_prob_maps': self.cam_prob_maps,
                                'lstm_pred': self.lstm_pred}, f)

            if not hasattr(self.data_loader, 'txt_stops'):
                if single_vis:
                    generate_attention_sequence_single(self.images, self.att_prob_maps, self.cam_prob_maps, self.lstm_pred, save_path)
                else:
                    generate_attention_sequence(self.images, self.att_prob_maps, self.cam_prob_maps, self.lstm_pred, save_path)
            else:
                # for topic_mdnet
                generate_attention_sequence2(self.images, self.att_prob_maps, self.cam_prob_maps, self.lstm_pred, save_path, sigma=7)
        ## clear up
        self.cnn_pred_list = []
        self.cnn_gt_list = []
        self.lstm_acc = 0
        self.lstm_pred = []
        self.att_prob_maps = []
        self.cam_prob_maps = []
        self.images = []

        return best_surfix

    def sampling(self, pred_logits, sample_max=True, tempature=1.0):
        """Comupute predicted word
            Input:
                pred_logits [batch_size*num_feature, max_subseq_len, vocab_size]
        """
        n, sl, v = pred_logits.shape
        num_feat = self.data_loader.num_feature
        if sample_max:
            preds = np.argmax(pred_logits, axis=2)
        else:
            preds = np.zeros(n*sl)
            # scale prediction by tempature
            scaled_logits = np.exp(pred_logits/tempature)
            flat_logit = np.reshape(pred_logits,[n*sl, v])

            for i in range(flat_logit.shape[0]):
                preds[i] = np.random.multinomial(1,flat_logit[i])
            assert(n % num_feat == 0) 
            preds = np.reshape(preds, [n, sl])

        preds = np.reshape(preds, [n/num_feat, num_feat, sl])

        return preds

    def save_to_mat(self, savepath):

        def unsampling(att, att_wh, thre=0.3, upscale=16, sigma=25, imsize=299):
            if len(att.shape) < 2:
                att = att / att.max() #TODO: pay attention to here the normalization is different 
                att[att<thre] = 0
                att = skimage.transform.pyramid_expand(att.reshape(att_wh, att_wh), upscale=upscale, sigma=sigma)
                att = skimage.transform.resize(att, [imsize, imsize])
                return att
            else:
                atts = np.zeros((att.shape[0], imsize, imsize), np.float32)
                for i in range(atts.shape[0]):
                    temp_att = att[i]
                    temp_att = temp_att / temp_att.max()
                    temp_att[temp_att<thre] = 0
                    temp_att = skimage.transform.pyramid_expand(temp_att.reshape(att_wh, att_wh), upscale=upscale, sigma=sigma)
                    temp_att = skimage.transform.resize(temp_att, [imsize, imsize])
                    atts[i] = temp_att
                return atts

        att_wh = int(np.sqrt(self.att_prob_maps[0].shape[1]))
        mat_out = dict()
        for i, att in enumerate(self.att_prob_maps):
            img_id = self.lstm_pred[i]['image_id']
            img = self.images[i]
            caption_raw = self.lstm_pred[i]['caption'].split(' ')
            caption = []
            assert(len(caption_raw) == att.shape[0])
            ## remove characters
            indices = []
            for idx, c in enumerate(caption_raw):
                if c != ':' and c != ".":
                    indices.append(idx)
                    caption.append(c)
            cam = unsampling(self.cam_prob_maps[i], att_wh) 

            caption = ' '.join(caption)
            att = unsampling(att[indices,:], att_wh)

            mat_out[img_id] = [att, cam, caption, img]

        print ('--> save attention map matfile at '+savepath+'_attention.mat')    
        io.savemat(savepath+'_attention.mat', mat_out)

    def to_demo_input(self, save_path, max_att_pixel_to_keep=2, original_imsize=500, normalize=False):
        # organize the data to the input format of the demo video for whole slide diagnosis
        print ('--> generate demo input ...')            

        att_wh = int(np.sqrt(self.att_prob_maps[0].shape[1]))
        stride = original_imsize/2
        # patch_coords_path = save_path+'patch_coords.json'
    
        # print ('can not find ' + patch_coords_path + ' generate it now')
        matfile = save_path + 'contours.mat'
        # with h5py.File(matfile, 'r') as f:
        #     print(f.keys())
        #     import pdb; pdb.set_trace()
        #     Boundaries = f['Boundaries']
        data = io.loadmat(matfile, squeeze_me=True,  struct_as_record=False)
        # # PatchPos = data['PatchPos']
        Boundaries = data['Boundaries'] # [y, x]

        region_contours = []
        # patch_coords = dict()

        # n = PatchPos.xrand.size
        # for c in range(n): # image id is in order from 1 to n
        #     # number begin from 1
        #     patch_coords[str(c+1)] = {'x': int(PatchPos.xrand[c]), 'y': int(PatchPos.yrand[c])}

        for c in range(len(Boundaries)):
            region = dict()
            contours = Boundaries[c]
            x = [int(m) for m in list(contours[:,1])]
            y = [int(m) for m in list(contours[:,0])]
            region['uid'] = c+1
            region['name'] = c+1
            region['points'] = {
                'x': x,
                'y': y
            }

            region_contours.append(region)
        
        json.dump({"Regions": region_contours}, open(save_path+'regions.json','w'), indent=2, sort_keys=True)
        print ('save regions.json ...')
            
        full_coords_list = []
        scale = original_imsize / att_wh
        for i, att in enumerate(self.att_prob_maps):
            
            img_id = self.lstm_pred[i]['image_id']
            caption = self.lstm_pred[i]['caption'].split(' ')
            time_len = self.lstm_pred[i]['step_len']
            x, y = img_id.split('_')[1:]
            patch_coord = {
                'x': int(x),
                'y': int(y) 
            }

            pos_dict = dict()
            series = list()

            if att.shape[0] <= 6:
                # new TopicMdenet version
                # expand attetion to each word time step
                # captialize raw caption
                expend_att = []
                expend_caption = caption.copy()
                cur_p = 0
                for kk, l in enumerate(time_len):
                    
                    expend_caption[cur_p] = expend_caption[cur_p].capitalize()
                    cur_p += l
                    expend_caption[cur_p-1] = expend_caption[cur_p-1] + '.'
                    expend_att += [att[kk]] * (l)
                    
                att = np.array(expend_att)
                caption = expend_caption

            # import pdb; pdb.set_trace()
            assert len(caption) == att.shape[0]

            # precomput alpha normalizetion over the largest algha of all time steps
            alpha_norm = np.zeros(att.shape[0], np.float32)
            for t in range(att.shape[0]):
                alpha = att[t].copy().reshape(-1) # for each time step
    
                assert(len(alpha.shape) == 1)
                idx = np.argsort(alpha,axis=0)[::-1] # sort in descending order
                idx = idx[0] # get the largest one
                alpha_norm[t] = alpha[idx]
            alpha_norm = alpha_norm.max()   
            
            for t in range(att.shape[0]): # for each time step
                if caption[t] == ':' or  caption[t] == ".":
                    continue
                series_t = dict() 
                
                alpha = att[t].copy().reshape(-1) # for each time step
                alpha = alpha / alpha_norm
            
                assert(len(alpha.shape) == 1)
                idx = np.argsort(alpha,axis=0)[::-1] # sort in descending order
                idx = idx[:max_att_pixel_to_keep]

                y, x = np.unravel_index(list(idx), (att_wh, att_wh))
                # convert to original size
                y = list(y * scale)
                x = list(x * scale)
                
                #desecent output
                
                # if caption[t+1] == '.':
                #     word = caption[t] + '.'
                #     if 'hg' in caption[t]:
                #         word = 'Suspect to be high grade carcinoma.'
                #     elif 'lg/punlmp' in caption[t]:
                #         word = 'Suspect to be low grade carcinoma.'
                #     elif 'normal' in caption[t]:
                #         word = 'Suspect to be normal.'
                # elif caption[t-1] == ':':
                #     word = caption[t].title()
                # else:
                word = caption[t]
                
                series_t['description'] = word
                series_t['coords'] = [{
                                    "x": min(x, original_imsize) + patch_coord['x']-stride, 
                                    "y": min(y, original_imsize) + patch_coord['y']-stride,
                                    'alpha': float(alpha[idx[c]])} 
                                    for c, (x, y) in enumerate(zip(x, y))
                                   ]
                series.append(copy.copy(series_t))

            ## DEPRECATED! interpolation makes the demo too slow.   
            # interpolate the alpha for each time step
            # for t in range(len(series)):
            
            #     alpha_pts = series[t]['coords']
            #     # for all points in each frame
            #     for pt_id in range(len(alpha_pts)):
            #         alpha = series[t]['coords'][pt_id]['alpha']
            #         # do interplation
            #         x = [0, 0.25, 0.5, 0.75, 1]
            #         y = [0, alpha/2, alpha, alpha/2, 0]
            #         f = scipy.interpolate.interp1d(x, y,kind='cubic')
            #         new_alpha_list = f(np.arange(0, 1, interp_interval))
          
            #         series[t]['coords'][pt_id]['alpha'] = list(new_alpha_list)


            pos_dict['series'] = series
            pos_dict['center'] = patch_coord
            pos_dict['img_id'] = img_id
            pos_dict['slide'] = img_id

            full_coords_list.append(copy.copy(pos_dict))
        
        label = save_path[save_path.find('type')+4]
        mapping = {'1': "Low grade papillary urthelial carcinoma", '0': 'Normal case', '2':  'High grade papillary urthelial carcinoma'}
        if label in mapping.keys():
            diagnosis = mapping[label]
        else:
            diagnosis = 'Unknown'
        res = {
            'attentions': full_coords_list,
            'diagnosis': diagnosis

        }
        
        json.dump(res, open(save_path+'/attentions.json','w'), indent=2, sort_keys=True)
            




