# -*- coding: utf-8 -*-

import os, sys
import numpy as np
import json


def get_coords(region):
    """ Arrange each region's contour coordinates.

    """

    name = region['name']
    points = region['points']
    num = len(points)
    coords = np.zeros((num, 2), np.float32)

    c = 0
    for i, poi in enumerate(points):
        if poi['x'] > 0 and poi['y'] > 0:
            coords[c][0] = poi['x']
            coords[c][1] = poi['y']
            c += 1
        else:
            continue
    # just keep valid points
    coords = coords[:c,:]

    return name, coords


def load_annotation(annotation_path):
    """ Load the region annotation to a dictionary.

    """

    if os.path.exists(annotation_path) == False:
        return None

    annotation_data = {}
    with open(annotation_path) as fp:
        annotation_data = json.load(fp)
    regions_list = annotation_data['Regions']

    region_dict = {}
    for cur_reg in regions_list:
        name, coords = get_coords(cur_reg)
        region_dict[name] = coords

    return region_dict
