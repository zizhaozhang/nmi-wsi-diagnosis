# vis
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.patheffects as PathEffects
import numpy as np
import scipy.io as io
import deepdish as dd
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score, precision_recall_curve, f1_score, precision_score, recall_score, confusion_matrix
import pdb, os
import six
from matplotlib import colors
import json, glob
from scipy.stats import mode

def get_sensitivity(y, score, label):
    # pdb.set_trace()

    all_pos = y == label
    tru_pos = (score == y) & (y == label)

    return float(np.sum(tru_pos)) /  np.sum(all_pos)

def get_specificity(y, score, label):
    all_neg = np.sum(y != label)
    tru_neg = ( score== y) & (y != label)

    return float(np.sum(tru_neg)) /  np.sum(all_neg)

def load_human_diagnosis():
    # go to /home/zizhaozhang/Dropbox/bladder_cancer_diagnosis/ for how to collect
    root = '.data/wsi/human_results/'
    target_slide = './data/wsi/selected_diagnosis_for_comparsion_100.json'
    lis = glob.glob(root + '*.json')
    print ('find {} human diagnosis results'.format(len(lis)))
    human_info_all = []
    human_names = []
    for l in lis:
        name = os.path.splitext(os.path.basename(l))[0]
        human_names.append(name)
        human_info = json.load(open(l))
        human_info_all.append(human_info)

    f = json.load(open(target_slide))
    used_slides = [a for a in f.keys()]
    human_preds = []
    for human_info in human_info_all:
        human_preds.append( np.array([human_info[k] - 1 for k in used_slides]) )
    print ('use {} selected diagnosis slides'.format(len(used_slides)))

    return human_preds, used_slides, human_names

def pr_evalation(y, scores, class_name={0: 'Low grade',1: 'High grade'}, save_path='./wsi_cls/', selected_slide=True):
    sns.set_style("white")
    num_class = len(class_name)

    y = np.array(y)
    scores = [np.reshape(a,newshape=(1, num_class)) for a in scores]
    scores = np.concatenate(scores,axis=0)

    # dd.io.save('./wsi_cls/auc_data.h5', {'y': y, 'scores': scores})
    # pdb.set_trace()

    ## load doctor accuracy
    human_predictions, used_slides, human_names = load_human_diagnosis()

    ''' minus 1 since it is binary test'''
    label_info = json.load(open('./wsi_cls/test_label_293.json'))
    groundtruth = np.array([label_info[k] - 1 for k in used_slides])
    assert(len(groundtruth) == len(y))

    print ('--> do precision-recall evaluation ...')
    # plt.ion()
    plt.figure(figsize=(7,3))

    average_mode = ''
    for pl in range(num_class):

        prec, rec, thresholds = precision_recall_curve(y==pl, scores[:,pl])
        f1 = f1_score(y==pl, scores[:,pl]>0.5)

        plt.subplot(121+pl)
        plt.plot(rec, prec,
                color='navy',linewidth=2,
                lw=2, label='Method (F1-score = %.3f)' % f1)
        plt.fill_between(rec, prec, step='post', alpha=0.2,
                    color='navy')

        avg_human_reca = []
        avg_human_prec = []
        for human_preds in human_predictions:
            human_prec = precision_score(groundtruth==pl, human_preds==pl)
            human_reca = recall_score(groundtruth==pl, human_preds==pl)
            human_f1 = f1_score(groundtruth==pl, human_preds==pl)
            avg_human_reca.append(human_reca)
            avg_human_prec.append(human_prec)
            plt.plot(human_reca, human_prec,
                    'ro',
                    label='Pathologists (%d)' % len(human_predictions))

        plt.plot(np.array(avg_human_reca).mean(0), np.array(avg_human_prec).mean(0),
                    'g*',
                    label='Average Pathologist')

        plt.xlim([0.0, max(rec)])
        plt.ylim([0.0, max(prec)+0.05])
        plt.xlabel('Recall',fontsize=6)
        plt.ylabel('Precision',fontsize=6)
        plt.title(class_name[pl], fontsize=6)
        plt.legend(loc="lower left", fontsize=6)

        # plt.savefig(save_path+'class_{}.pdf'.format(pl), bbox_inches='tight')

    print ('save precision_recall at', save_path+'precision_recall.pdf')
    plt.savefig(save_path+'precision_recall.pdf', bbox_inches='tight')


COLORS = [a[0] for a in list(six.iteritems(colors.cnames))]

def auc_evalation(oy, oscores, class_name={0: 'Low grade',1: 'High grade'}, name_list=None, save_path=None, selected_slide=True):
    sns.set_style("white")
    num_class = len(class_name)
    oy = np.array(oy)
    oscores = [np.reshape(a,newshape=(1, num_class)) for a in oscores]
    oscores = np.concatenate(oscores,axis=0)

    if save_path is not None:
        #load doctor performance
        print('--> AUC evaluation ...')
        label_info = json.load(open('./data/wsi/test_label_293.json'))
        if selected_slide:
            human_predictions, used_slides, human_names=load_human_diagnosis()
        else:
            used_slides = [a for a in label_info.keys()]

        groundtruth = np.array([label_info[k] - 1 for k in used_slides])

        y = np.zeros(len(used_slides))
        scores = np.zeros((len(used_slides), oscores.shape[1]) )
        for i in range(len(used_slides)):
            ind = name_list.index(used_slides[i])
            y[i] = oy[ind]
            scores[i] = oscores[ind]

        assert(len(groundtruth) == len(y))

        # plt.ion()
        plt.figure(figsize=(7,3))
        for pl in range(num_class):
            FPR, TPR, thresholds = roc_curve(y==pl, scores[:,pl], drop_intermediate=False)
            auc = roc_auc_score(y==pl, scores[:,pl])

            sensitivity = TPR
            specificity = 1 - FPR

            # interplate the values if not full in [0,1]
            if sensitivity[0] != 0:
                inter_p = int(sensitivity[0] // (sensitivity[1] - sensitivity[0]))
                sensitivity_p = [0] + [sensitivity[0]/a for a in range(2, 2+inter_p)[::-1]]
                sensitivity = sensitivity_p + sensitivity.tolist()
                specificity = [1] * len(sensitivity_p) + specificity.tolist()
            plt.subplot(121+pl)
            plt.fill_between(sensitivity, specificity, step='post', alpha=0.2,
                        color='navy')
            plt.plot(sensitivity, specificity,
                    color='navy',linewidth=3,
                    label='Method AUC = %.2f' % auc)

            # human
            avg_human_sens = []
            avg_huamn_spec = []
            for h, human_preds in enumerate(human_predictions):

                # confusion matrix
                # conf  = confusion_matrix(groundtruth, human_preds)
                # human_acc = accuracy_score(groundtruth, human_preds)
                # print ('-'*50)
                # print ('Overal accuracy (human {}): {} '.format(h, human_acc))
                # print ('Confusion matrix: ')
                # print (conf)
                # print('-' * 50)

                human_sensitivity = get_sensitivity(groundtruth, human_preds, pl)
                human_specificity = get_specificity(groundtruth, human_preds, pl)
                human_auc = roc_auc_score(groundtruth==pl, human_preds==pl,)
                avg_human_sens.append(human_sensitivity)
                avg_huamn_spec.append(human_specificity)
                plt.plot(human_sensitivity, human_specificity,
                        'ro', )
                        # # for testing
                        # color=COLORS[int(human_names[h][0])*5],
                        # label='%s' % human_names[h])


            plt.plot(0, 0,
                    'ro', markersize=0, marker='h',
                    label='Pathologists (%d)' % len(human_predictions))
            plt.plot(np.array(avg_human_sens).mean(0), np.array(avg_huamn_spec).mean(0),
                    'gH', marker='H',
                    label='Average Pathologist')

            plt.xlim([0.0, 1.05])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Sensitivity',fontsize=10)
            plt.ylabel('Specificity',fontsize=10)
            plt.title(class_name[pl], fontsize=10)
            plt.legend(loc="lower left", fontsize=10)
            plt.xticks([a/4 for a in np.arange(5)], ('0', '', '', '', '1'))

            # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

        print ('save auc', save_path+'auc_{:0.2f}.pdf'.format(auc))
        plt.savefig(save_path+'auc_{:0.2f}.pdf'.format(auc), bbox_inches='tight')

    else:
        # just do auc evaluation
        pl = 0
        auc = roc_auc_score(oy == pl, oscores[:, pl])
    return auc

def scatter(x, labels, name, f=None, ax=None, isslide=True):
    num_label = np.max(labels) +1

    palettes = [np.array(sns.light_palette(col, n_colors=10, reverse=True)) for col in ['blue', 'red']]
    markers = ["o",'d', 's', '^', '8','p', '+', 'h', '2','1']
    # for p in palettes: sns.palplot(p)
    labels = labels.astype(np.int)
    # We create a scatter plot.
    if f is None:
        f = plt.figure(figsize=(8, 8))
        ax = plt.subplot(aspect='equal')
    if isslide:
        marker_size = 200
        maker = ["v", '^']
        sat = 4
    else:
        marker_size = 50
        maker = "o"
        sat = 5
    clasmap = {
        0: 'low grade',
        1: 'high grade'
    }
    for i in range(num_label):
        idx = np.where(labels==i)[0]
        pal = palettes[i]
        # for s in range(idx.size):
        sc = ax.scatter(x[idx,0], x[idx,1], lw=1, s=marker_size,
                    marker=maker[i],
                    edgecolor='white',
                    c=pal[sat], label='Slide (class {})'.format(i+1))

        sc = ax.scatter(np.mean(x[idx,0]), np.mean(x[idx,1]), lw=1, s=marker_size*2,
                    marker='X',
                    c=pal[0], label='Center (class {})'.format(i+1))

    # plt.xlim(-25, 25)
    # plt.ylim(-25, 25)
    plt.title(name)
    ax.axis('off')
    ax.axis('tight')
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    return f, ax

def tsne_visualization(set_layer_outputs):
    """
    set_layer_outputs: [item, ...]. Each item is a list containing [f_3076,f_512,f_128,f_2]
    """
    print('tsne visualization ... ')
    sns.set_style('darkgrid')
    sns.set_palette('muted')
    sns.set_context("notebook", font_scale=1.5,
                    rc={"lines.linewidth": 2.5})
    # organize data by feature type
    num_of_feat = 4
    for k in range(0, num_of_feat):
        feats = [s[k] for s in set_layer_outputs] # feat k for all data
        label = []
        # image_idx = []
        # for i in range(len(set_layer_outputs)):
        #     image_idx += [i for c in range(feats[i].shape[0])]
        for s in set_layer_outputs:
            label.extend(np.argmax(s[-1],1).tolist())
        X = np.concatenate(feats, axis=0)
        xshape = X.shape
        print (' shape of X for vis: ' + str(X.shape))
        # model = TSNE(n_components=2, perplexity=50, n_iter=1000, learning_rate=10, random_state=1234) # good results
        model = TSNE(n_components=2, perplexity=70, n_iter=3000, learning_rate=1, random_state=1234) #
        np.set_printoptions(suppress=True)
        tsne_data = model.fit_transform(X)


        # get the mean of each slide samples
        selected_tsne_data = []
        selected_label = []
        for i in range(len(label) // 10):
            s = i * 10
            e = s + 10
            selected_tsne_data.append(tsne_data[s:e,:].mean(0)) # get the mean of samples
            selected_label.append(mode(label[s:e], axis=0)[0][0])
        selected_tsne_data = np.array(selected_tsne_data)
        selected_label = np.array(selected_label)

        f, ax = scatter(selected_tsne_data, selected_label, name='Feature '+str(xshape[1]))
        f.savefig('Feature '+str(xshape[1])+'_slide.pdf', bbox_inches='tight')

        # f2, ax = scatter(tsne_data, np.array(label), name='Feature '+str(xshape[1]))
        # f2.savefig('Feature '+str(xshape[1])+'_sample.pdf', bbox_inches='tight')

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, path):

        self.path = path
        self.history = {}
        self.avg = 0
        self.reset_save()


    def reset_save(self, epoch=0):
        self.history[epoch] = self.avg
        json.dump(self.history, open(self.path,'w'), indent=2)

        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
