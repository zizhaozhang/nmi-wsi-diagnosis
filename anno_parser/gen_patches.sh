#!/bin/bash

# # generate high grade patches in train, namely in category "2"
# python gen_patches.py --anno_type "Pos" --dset "train" --slide_id "1_00061_sub0"

# # generate low grade patches in val, namely in category "1"
# python gen_patches.py --anno_type "Pos" --dset "test" --slide_id "1_00161_sub0"

# generate normal patches in train, namely in category "3"
python gen_patches.py --anno_type "Neg" --dset "train" --slide_id "1_00066_sub0"
