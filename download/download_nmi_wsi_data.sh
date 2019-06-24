#! /bin/bash

root_dir=nmi-wsi-diagnosis
mkdir $root_dir
echo "Creating root directory: ${root_dir}"
cd $root_dir
echo $(pwd -P)

mkdir Report
cd Report
echo "Downloading Report now..."
wget https://www.dropbox.com/s/40p9137riixv36e/class.json
wget https://www.dropbox.com/s/vvrimsow0zmr4zn/test_annotation.json
wget https://www.dropbox.com/s/5t0pgmjzshjz13o/train_annotation.json
wget https://www.dropbox.com/s/v558k4verlicw16/Images.zip
cd ..

mkdir Slide
cd Slide
echo "Downloading Slide now..."
wget https://www.dropbox.com/s/wmwwayn5s2zve77/class_slide.json
wget https://www.dropbox.com/s/vfvy21drom261jl/Thumbnails.zip
wget https://www.dropbox.com/s/1kpu6xv27s5ksvv/RegionAnnotation.zip
wget https://www.dropbox.com/s/legnlzicbgyosnr/human_results.zip
wget https://www.dropbox.com/s/ri02tutxecqk85t/selected_diagnosis_for_comparsion_100_partial.json
wget https://www.dropbox.com/s/e2suyskcfky115t/selected_diagnosis_for_comparsion_100.json
wget https://www.dropbox.com/s/cynm50cgpysilms/train_diagnosis_partial.json
wget https://www.dropbox.com/s/j4jr8tymv9rlpwz/train_diagnosis.json
wget https://www.dropbox.com/s/e9me5f4yzfrp5ai/val_test_diagnosis_partial.json
wget https://www.dropbox.com/s/lfdgv9my5vjtge7/val_test_diagnosis.json
mkdir Img
cd Img
echo "Downloading slides now, it takes some time"
wget -i ../../../nmi_wsi_slide_list.txt
cd ../../../
echo "Download finish!!!"

