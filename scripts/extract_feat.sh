#!/usr/bin/env bash

model_name='checkpoint_49_28.83' # change this
name='segmentor'

root='checkpoints/seg_test_slides/'
CUDA_VISIBLE_DEVICES=${device} python extract_feat.py \
        --data_root ${root} \
        --folder ${name} \
        --save_name ${name}_inception_feat \
        --model_name $model_name \
        --slide_label 'data/wsi/val_test_diagnosis.json'

root='checkpoints/seg_train_slides/'
CUDA_VISIBLE_DEVICES=${device} python extract_feat.py \
        --data_root ${root} \
        --folder ${name} \
        --save_name ${name}_inception_feat \
        --model_name $model_name \
        --slide_label 'data/wsi/train_diagnosis.json'
