# Slide Annotation Loading and Patch Generation
### Slide Annotation Visualization
```
$ python overlay_anno.py --bladder_dir "../data" --anno_type "Pos" --slide_id "1_00061_sub0" --slide_level 3
```

<img src="./pos_1_00061_sub0.png" width="280" height="300" alt="Overlay Demo">

#### Parameters
--bladder_dir: the root of bladder whole slide image dataset     
--anno_type: the region annotation type, includes "Pos" and "Neg"   
--slide_id: the id of a bladder slide   
--slide_level: the level in slide pyramidal storage structure to load


## Patch Generation for Segmentation and Classification
#### Patch label design
* When the patch is cropped based on "Pos" annotated contours, if the slide is diagnosed as low grade (LG), the patch label would be "1", and if the slide is diagnosed as high grade (HG), the label would be "2". All slides' diagnosis information can be found in json files along with slide files. When the patch is cropped from "Neg" annotated contour, the patch label would be "3". Details can be found in the original paper also.

#### Segmentation mask design
* According to the Methods section of the paper, quoted, "the II-Image data set is used to train s-net. Because the tumour region is partially annotated, unannotated region pixels have unknown classes. To bypass this problem, during network training we compute the losses of s-net only for annotated region pixels, while ignoring unannotated pixels". 
* For patches cropped from "Pos" annotated contours, those pixels inside contours would be given a value of `255`, namely `positive`, no matter the slide diagnosis is low grade or high grade. While those pixels outside contours would be given value of `44`, namely `ignore`.
* For patches cropped from "Neg" annotated contours, those pixels inside contours would be given value of `155`, namely `negative`. While pixels outside contours would be given value of `44` to stand for `ignore`.
* See ```data_loader()``` in segmentation/data_gen.py for more details.

### Patch Generation Demo for model training
```
python gen_patches.py --anno_type "Neg" --dset "train" --slide_id "1_00066_sub0"
```
