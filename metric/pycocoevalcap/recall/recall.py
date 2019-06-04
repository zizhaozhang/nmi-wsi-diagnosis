import numpy as np
from sklearn.metrics import recall_score, confusion_matrix,precision_score, accuracy_score

class Recall:
    def __init__(self):
        self.class_discrete = {'normal':1, 'insufficient information':4, 
                             'low grade': 2 ,'high grade': 3}
    
    def extract_keyword(self, sent):

        temp_words = sent.split() # using the first refernece is fine because we have all references thas the same conclusion
        if len(temp_words) < 1:
            return None

        if temp_words[-1] == 'information':
            if len(temp_words) >=2 and temp_words[-2] == 'insufficient':
                keyword = 'insufficient information'
            else:
                keyword = None
        elif temp_words[-1] not in self.class_discrete.keys():
            keyword = None
        else:    
            keyword = temp_words[-1]
        # assert (keyword == 'insufficient information' or 
        #         keyword == 'hg' or 
        #         keyword == 'lg/punlmp' or     
        #         keyword == 'normal')
        return keyword

    def compute_score(self, gts, refs):
        assert(gts.keys() == refs.keys())
        imgIds = gts.keys()
        total_relv = 0.0
        tp = 0.0
        label_gt = [0 for k in imgIds]
        label_pred = [0 for k in imgIds]
        for i, id in enumerate(imgIds):
            key_gt = self.extract_keyword(gts[id][0]) #using the first reference
            label_gt[i] = self.class_discrete[key_gt]
            key_ref = self.extract_keyword(refs[id][0])
            if key_ref == None: 
                label_pred[i] = -1 
                continue
            label_pred[i]= self.class_discrete[key_ref]

        tmp = np.array(label_pred)
        if len(tmp[tmp==-1]) > 0:
            print ('Recall Evaluation Warning: Existing irrelevant values')
            for k in range(len(label_pred)):
                if label_pred[k] == -1:
                    label_pred[k] = max(1,label_gt[k] - 1) # make it different from label_gt

        cfm = confusion_matrix(label_gt, label_pred)
        recall = recall_score(label_gt, label_pred, average='weighted')
        precision = precision_score(label_gt, label_pred, average='weighted')
        acc = accuracy_score(label_gt, label_pred)
        score = (precision, recall, acc)
        # print ('-------------------------------------------------------------')
        print ('LSTM confusion matrix:')
        print (cfm)
        # print ('recall: ', recall)
        # print ('precision: ', precision)
        # print ('accuracy: ', acc)
        scores = [score for i in range(len(imgIds))]
        return score, scores
    def method(self):
        return 'Recall'