#!/usr/bin/env bash
# change feat_data_name and name together 
root='checkpoints/diagnosis'
# where IV-Diagnosis saved
name='test_mlp_cnnfeat_[0123]_model_1024_256_0.2droprate_v2'

mkdir -v $root/$name
CUDA_VISIBLE_DEVICES=${device_id} python mlp_train.py \
       --log_dir $root/$name \
       --feat_data_name 'segmentor_drop0.0_ignorew0.5_checkpoint_3_86.80_cnnfeat' \
       --lr_decay_epoch 50 \
       --tot_epoch 500 \
       --test_per_epoch 5 \
       --drop_rate 0.2 \
       --feat_comb 1 \
       --learning_rate 0.001 \
       --lr_decay 0.1 \
       --use_cls_weight 1