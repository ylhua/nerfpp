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

import os
import cv2

i  =1
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
import cv2
import imageio
import os

def label_img_to_color(img):
    # label_to_color = {
    #     0: [40, 40,40],
    #     1: [230, 40,40],
    #     2: [40, 230, 40],
    #     3: [40,40,230],
    #     4: [190,153,153],
    #     5: [153,53,153],
    #     6: [250,170, 30]
    #     }

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
    semantic_file = '/media/linger/ubuntu/null_max/label_manual/nerfplusplus/nuscenes/test/semantic'
    img_file = '/media/linger/ubuntu/null_max/label_manual/nerfplusplus/nuscenes/test/rgb/'
    save_file = '/media/linger/ubuntu/null_max/label_manual/nerfplusplus/nuscenes/test/seg_vis/'
    blend_save_file = '/media/linger/ubuntu/null_max/label_manual/nerfplusplus/nuscenes/test/blend_vis/'

    list = os.listdir(semantic_file)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('/media/linger/ubuntu/null_max/label_manual/nerfplusplus/nuscenes/test/output.avi', fourcc, 20.0, (1600, 900))
    video_vis = []
    for i in range(0, 4):
        path = os.path.join(semantic_file, list[i])


        if os.path.splitext(path)[1] != '.png':
            continue

        label_ori = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        label_vis_img = label_img_to_color(label_ori)
        img_ori_dir = os.path.join(img_file, list[i].split('.')[0] + '.jpg')
        img_ori = np.zeros_like(label_vis_img)
        img = cv2.imread(img_ori_dir, cv2.IMREAD_GRAYSCALE)
        img_ori[:, :, 0] = img
        img_ori[:, :, 1] = img
        img_ori[:, :, 2] = img
        img_vis = img_ori * 0.5 + label_vis_img * 0.5

        cv2.imwrite(os.path.join(save_file, list[i]), label_vis_img)
        cv2.imwrite(os.path.join(blend_save_file, list[i]), img_vis)

        out.write(img_vis)
        video_vis.append(img_vis)

    imageio.mimwrite('F:/output.mp4', video_vis, fps=30, quality=8)

if __name__ == '__main__':
    main()


