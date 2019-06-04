#!/usr/bin/env bash
root='checkpoints/diagnosis'
name='test_mlp_cnnfeat_[0123]_model_1024_256_0.2droprate_v2'
echo 'TEST '$name
CUDA_VISIBLE_DEVICES=${device_id} python mlp_test.py \
       --log_dir $root/$name \
       --feat_root './checkpoints' \
       --feat_data_name 'segmentor_inception_feat_checkpoint_49_28.83_cnnfeat_wloc' \
       --feat_comb 1 \
       --load_model_from $root/$name/'model_5_0.979' \
       | tee $root/$name/'test_log.txt'
