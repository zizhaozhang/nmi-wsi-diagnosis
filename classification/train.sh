
name='trained_model'
logpath='../checkpoints/classification/'$name
mkdir $logpath
CUDA_VISIBLE_DEVICES=$device python train.py \
                --batch_size 64 \
                --checkpoint_path $name \
                --learning_rate 0.0005 \
                --lr_decay 0.9 \
                --lr_decay_epoch 1 \
                --optim 'rmsp' \
                --use_cls_weight 2 \
                --iter_epoch_ratio 1 \
                --data_path '../data/classification/'
