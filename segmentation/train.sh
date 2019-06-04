
name='segmentor'
CUDA_VISIBLE_DEVICES=$device python train.py \
                --batch_size 32 \
                --checkpoint_path $name \
                --learning_rate 0.0001 \
                --lr_decay 0.9 \
                --epoch 6 \
                --lr_decay_epoch 1 \
                --optim 'adam' \
                --iter_epoch_ratio 0.3 \
                --drop_rate 0.0 \
