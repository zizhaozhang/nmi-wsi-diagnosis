
name='segmentor'
logpath='../checkpoints/segmentation/'$name

# save ROI in $res_dir
CUDA_VISIBLE_DEVICES=$device python seg_wsi.py \
               --batch_size 12 \
               --checkpoint_path $name \
               --eval \
               --wsi_dir '../data/wsi/'${split}'_slides/' \
               --slide_level 0 \
               --threshold 0.5 \
               --imSize 512 \
               --end $end \
               --start $start \
               --res_dir '../checkpoints/seg_'${split}'_slides/'$name \
               --load_from_checkpoint $logpath/'checkpoint_5.h5' # change this accordingly
