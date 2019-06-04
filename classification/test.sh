
name='trained_model' # best
logpath='../checkpoints/classification/'$name
CUDA_VISIBLE_DEVICES=$device python test.py \
               --batch_size 64 \
               --checkpoint_path $name \
               --eval \
               --data_path '../data/classification/' \
               --load_from_checkpoint $logpath'/checkpoint_49_28.83.h5' # modify this
                # this is currently the best results
