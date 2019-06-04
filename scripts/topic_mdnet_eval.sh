#!/usr/bin/env bash

log_dir='checkpoints/report/TopicMDNet/'

CUDA_VISIBLE_DEVICES=${device} python topic_mdnet_test.py \
        --dataset_dir='data/report' \
        --log_dir=${log_dir} \
        --load_from_checkpoint=${log_dir}"model1.0_BEST0.29-367" \
        --model='TopicMDNet8' \
        --sample_max=True \
        --test_mode=True \
        --batch_size 50
