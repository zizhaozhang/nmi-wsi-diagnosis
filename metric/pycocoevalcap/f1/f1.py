# compute f1 score for bladder image dataset
# zizhao @ UF

import numpy as np

class F1:

  def __init__(self):
    pass
  def compute_frequency(self, gts):
    freq = dict()
    with open('../../datasets/bladder/data_organized_multiref/bladder_vocab.txt','r') as f:
      for t in f.readlines():
        freq[t.strip().split(',')[1]] = 0
    
    for key in gts.keys():
      for ref in gts[key]:
        ref = ref.split()
        for word in ref:
            freq[word] = freq[word] + 1

    self.freq_norm = max(freq.values())
    self.weight = freq.copy()
    
    for i in self.weight.keys():
      if freq[i] == 0:
        self.weight[i] == 0
      else:
        # self.weight[i] = np.log(self.freq_norm) / max(np.log(freq[i]), 1)
        self.weight[i] = self.freq_norm / max(freq[i], 1)

  def compute_score(self, gts, res):
    
    assert(gts.keys() == res.keys())
    self.compute_frequency(gts)

    imgIds = gts.keys()
    scores = list()

    for id in imgIds:
      # Sanity check.
      # assert(type(hypo) is list)
      # assert(len(hypo) == 1)
      # assert(type(ref) is list)
      # assert(len(ref) > 0)

      hypo = res[id][0]
      f1 = 0
      for i in range(len(gts[id])):
        
        ref  = gts[id][i] # we have multiple references
        ss = self.get_score(hypo, ref) 
        f1 += self.get_score(hypo, ref)
        
      scores.append(f1 / len(gts[id]))
    score = sum(scores) / len(scores)

    return score, scores

  def get_score(self, hypo, ref):

    hypo = hypo.split()
    ref = ref.split()
    
    basic_scores = dict({'tp':0.0,'fp':0.0,'fn':0.0}) # [true positive, false positive, false negative]

    try:
      for gram in hypo:
        if gram in ref:
          basic_scores['tp'] += self.weight[gram] # true positive
        else:
          basic_scores['fp'] += self.weight[gram] # false positive
      for gram in ref:
        if gram not in hypo:
          basic_scores['fn'] += self.weight[gram] # false negative
    except Exception, e:
      print (e)
      print (gram)
    
    # print ('tp: '+str(basic_scores['tp'])+' fp: '+str(basic_scores['fp'])+' fn: '+str(basic_scores['fn']))

    precision = basic_scores['tp'] / (basic_scores['tp'] + basic_scores['fp'])
    recall = basic_scores['tp'] / (basic_scores['tp']+ basic_scores['fn'])

    # print ('precision: ', precision)
    # print ('recall: ', recall)
    
    # print ('precision: '+str(precision)+' recall: '+str(recall))
    # assert(precision <= 1 and precision > 0)
    # assert(recall <= 1 and precision > 0)
    if precision == 0 and recall == 0:
      f1 = 0
    else:
      f1 = 2 * (precision * recall) / (precision + recall)
    
    return f1
  
  def method(self):
    return 'F1 score'