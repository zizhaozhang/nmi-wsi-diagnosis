
name='segmentor'
logpath='../checkpoints/segmentation/'$name
checkpointname='checkpoint_5.h5'
CUDA_VISIBLE_DEVICES=$device python train.py \
               --batch_size 24 \
               --checkpoint_path $name \
               --iter_epoch_ratio 0.3 \
               --eval \
               --load_from_checkpoint $logpath/$checkpointname
