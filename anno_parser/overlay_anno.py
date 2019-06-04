# -*- coding: utf-8 -*-

import os, sys
import numpy as np
import argparse
import openslide
import cv2
import matplotlib.pyplot as plt
from load_anno import load_annotation


def display_annotation(slide_path, annotation_dict, slide_level):
    """ Draw the annotation contour onto the slide image on a specific level.

    """

    slide_head = openslide.OpenSlide(slide_path)
    slide_name = os.path.basename(slide_path)
    if slide_level < 0 or slide_level >= slide_head.level_count:
        print("level {} not availabel in {}".format(slide_level, slide_name))
        sys.exit()

    # load slide image
    slide_img = slide_head.read_region((0, 0), slide_level, slide_head.level_dimensions[slide_level])
    slide_img = np.asarray(slide_img)[:, :, :3]
    slide_img =  np.ascontiguousarray(slide_img, dtype=np.uint8)

    # draw annotation on the slide image
    for cur_reg in annotation_dict:
        coords = (annotation_dict[cur_reg] / slide_head.level_downsamples[slide_level]).astype(np.int32)
        cv_coords = np.expand_dims(coords, axis=1)
        slide_img = cv2.drawContours(slide_img, [cv_coords], 0, (0, 0, 255), 5)

    # display overlaid annotation slide image
    plt.imshow(slide_img)
    plt.axis('off')
    plt.title("Annotations overlaid on {}".format(slide_name))
    plt.tight_layout()
    plt.show()


def set_args():
    parser = argparse.ArgumentParser(description='Bladder slide annotation loading and visulization')
    parser.add_argument('--bladder_dir', type=str, default="./data",
                        help="folder that contains bladder data") # change based on your bladder data location
    parser.add_argument('--anno_type',   type=str, default="Pos", choices=['Pos', 'Neg'])
    parser.add_argument('--slide_id',    type=str, default="1_00061_sub0")
    parser.add_argument('--slide_level', type=int, default=3)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = set_args()

    # locate annotation path
    anno_path = os.path.join(args.bladder_dir, "Slide", "RegionAnnotation",
                             args.anno_type, args.slide_id, "annotations.json")
    if not os.path.exists(anno_path):
        print("No {} annotation for slide {}".format(args.anno_type, args.slide_id))
        sys.exit()

    # load annotation
    annotation_dict = load_annotation(anno_path)

    # find slide path (slide filename extension can be .svs or .tiff)
    slide_dir = os.path.join(args.bladder_dir, "Slide", "Img")
    if os.path.exists(os.path.join(slide_dir, args.slide_id+".svs")):
        slide_path = os.path.join(slide_dir, args.slide_id+".svs")
    elif os.path.exists(os.path.join(slide_dir, args.slide_id+".tiff")):
        slide_path = os.path.join(slide_dir, args.slide_id+".tiff")
    else:
        print("Slide {} not exist".format(args.slide_id))
        sys.exit()

    # overlay the annotation on the slide and display
    display_annotation(slide_path, annotation_dict, args.slide_level)
