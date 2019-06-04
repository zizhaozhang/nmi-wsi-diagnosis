#!/usr/bin/env bash
# root='/home/zizhaozhang/work/tf_tmp_checkpoints/checkpoint_mdnet6_softmax'
# root='/home/zizhaozhang/work/tf_tmp_checkpoints/checkpoint_mdnet8_0.9decay/'
# target='104818_sub1_type2'

# CUDA_VISIBLE_DEVICES=${device_id} python mdnet_test.py \
#         --dataset_dir='/home/zizhao/work/deepp2/datasets/bladder/data_organized_5fold2' \
#         --log_dir=${root} \
#         --load_from_checkpoint=${root}"model3.5_BEST(0.87)-7006" \
#         --model='MDNet8' \
#         --end2end=True \
#         --test_mode=True \
#         --test_image_dir="/media/zizhaozhang/data2/TCGA/UFWSItest/SampledData/"${target} \
#         --batch_size=1 \

log_dir='/home/zizhao/work2/mdnet_checkpoints/report/final_checkpoint_TopicMDNet8_ftat5_aug/'
# target='104818_sub1_type2'
target='104837_sub0_type1'
CUDA_VISIBLE_DEVICES=${device} python topic_mdnet_deploy.py \
        --dataset_dir='/home/zizhao/work2/dataset/bladder/report' \
        --log_dir=${log_dir} \
        --load_from_checkpoint=${log_dir}"model15.5-5703" \
        --model='TopicMDNet8' \
        --sample_max=True \
        --test_mode=True \
        --test_image_dir="./experiment/demo/TestSlides/SampleData/"${target}/ \
        --batch_size=1 \
        --to_demo_input=True