import json

root = '/media/zizhaozhang/DataArchiveZizhao/Bladder/merged/Report/'
data1 = json.load(open(root+'test_annotation.json','r'))
## data2 = json.load(open(root+'train_annotation.json','r'))
## data = {**data1, **data2}
data = data1
allres = []
for c, (k,v) in enumerate(data.items()):
  
    caption = []
    for capo in v['caption']:
        cap = capo.replace('. ','.').lower().rstrip('.')

        # !!! for topicmdnet
        caps = cap.split('.')
        # import pdb; pdb.set_trace()
        assert len(caps) == 6, capo
        for i in range(len(caps)):
            if i != len(caps)-1 and 'insufficient' in caps[i]: # not the conclusion
                caps[i] = ''
        cap = ' '.join([a for a in caps if a != ''])

        
        caption.append(cap)

    res = {
        'image_id': k,
        'id': c,
        'caption': caption
    }

    allres.append(res)


json.dump(allres, open('test_annotation_striped_topicmdnet.json','w'))
