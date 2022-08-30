import argparse
import json
import os
import os.path as osp
import warnings
import copy

import numpy as np
import PIL.Image
import yaml
import cv2

from labelme import utils

NAME_LABEL_MAP = {
    '_background_': 0,
    "road": 1,
    "laneline": 2,
    "leadline": 3,
    "road_sign": 4,
    "vehicle": 5,
    "person": 6
}

label_to_color = {
        0: [0, 0, 0],
        1: [244, 35, 232],
        2: [70, 70, 70],
        3: [102, 102, 156],
        4: [190, 153, 153],
        5: [153, 153, 153],
        6: [250, 170, 30]
        }

def label_img_to_color(img):
    label_to_color = {
        0: [40, 40,40],
        1: [230, 40,40],
        2: [40, 230, 40],
        3: [40,40,230],
        4: [190,153,153],
        5: [153,53,153],
        6: [250,170, 30]
        }

    img_height, img_width, _ = img.shape

    img_color = np.zeros((img_height, img_width, 3))
    for row in range(img_height):
        for col in range(img_width):
            label = img[row, col, 0]

            img_color[row, col] = np.array(label_to_color[label])

    return img_color

def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument('json_file')
    # parser.add_argument('-o', '--out', default=None)
    # args = parser.parse_args()
    #
    # json_file = args.json_file
    img_file = '/media/linger/ubuntu/null_max/label_manual/nerfplusplus/nuscenes/train/semantic/'
    save_file = '/media/linger/ubuntu/null_max/label_manual/nerfplusplus/nuscenes/train/seg_vis/'

    list = os.listdir(img_file)
    for i in range(0, len(list)):
        path = os.path.join(img_file, list[i])

        if os.path.splitext(path)[1] != '.png':
            continue

        label_ori = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        label_vis_img = label_img_to_color(label_ori)
        cv2.imwrite(os.path.join(save_file, list[i]), label_vis_img)


if __name__ == '__main__':
    main()


