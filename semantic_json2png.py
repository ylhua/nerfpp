# import argparse
# import json
# import os
# import os.path as osp
# import warnings
# import copy
# import numpy as np
# import PIL.Image
# import yaml
# from labelme import utils
#
# def main():
#     # parser = argparse.ArgumentParser()
#     # parser.add_argument('json_file')   # 标注文件json所在的文件夹
#     # parser.add_argument('-o', '--out', default=None)
#     # args = parser.parse_args()
#     #
#     # json_file = args.json_file
#     json_file = 'F:/null_max/label_manual/seg_res/CAM_FRONT/'
#
#     list = os.listdir(json_file)   # 获取json文件列表
#     for i in range(0, len(list)):
#         path = os.path.join(json_file, list[i])  # 获取每个json文件的绝对路径
#         filename = list[i][:-5]       # 提取出.json前的字符作为文件名，以便后续保存Label图片的时候使用
#         extension = list[i][-4:]
#         if extension == 'json':
#             if os.path.isfile(path):
#                 data = json.load(open(path))
#                 img = utils.image.img_b64_to_arr(data['imageData'])  # 根据'imageData'字段的字符可以得到原图像
#                 # lbl为label图片（标注的地方用类别名对应的数字来标，其他为0）lbl_names为label名和数字的对应关系字典
#                 lbl, lbl_names = utils.shape.labelme_shapes_to_label(img.shape, data['shapes'])   # data['shapes']是json文件中记录着标注的位置及label等信息的字段
#
#                 #captions = ['%d: %s' % (l, name) for l, name in enumerate(lbl_names)]
#                 #lbl_viz = utils.draw.draw_label(lbl, img, captions)
#                 out_dir = osp.basename(list[i])[:-5]+'_json'
#                 out_dir = osp.join(osp.dirname(list[i]), out_dir)
#                 out_dir = json_file + 'output/' + out_dir
#                 if not osp.exists(out_dir):
#                     os.mkdir(out_dir)
#
#                 PIL.Image.fromarray(img).save(osp.join(out_dir, '{}_source.png'.format(filename)))
#                 PIL.Image.fromarray(lbl).save(osp.join(out_dir, '{}_mask.png'.format(filename)))
#                 #PIL.Image.fromarray(lbl_viz).save(osp.join(out_dir, '{}_viz.jpg'.format(filename)))
#
#                 with open(osp.join(out_dir, 'label_names.txt'), 'w') as f:
#                     for lbl_name in lbl_names:
#                         f.write(lbl_name + '\n')
#
#                 warnings.warn('info.yaml is being replaced by label_names.txt')
#                 info = dict(label_names=lbl_names)
#                 with open(osp.join(out_dir, 'info.yaml'), 'w') as f:
#                     yaml.safe_dump(info, f, default_flow_style=False)
#
#                 print('Saved to: %s' % out_dir)
#
#
# if __name__ == '__main__':
#     main()
#


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

def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument('json_file')
    # parser.add_argument('-o', '--out', default=None)
    # args = parser.parse_args()
    #
    # json_file = args.json_file
    json_file = '/media/linger/ubuntu/null_max/label_manual/nerfplusplus/semantic/CAM_FRONT/'

    list = os.listdir(json_file)
    for i in range(0, len(list)):
        path = os.path.join(json_file, list[i])

        if os.path.splitext(path)[1] != '.json':
            continue

        filename = list[i][:-5]       # .json
        if os.path.isfile(path):
            data = json.load(open(path))
            img = utils.image.img_b64_to_arr(data['imageData'])
            lbl, lbl_names = utils.shape.labelme_shapes_to_label(img.shape, data['shapes'])  # labelme_shapes_to_label

            # modify labels according to NAME_LABEL_MAP
            lbl_tmp = copy.copy(lbl)
            # lbl_tmp = np.zeros_like(lbl)
            for key_name in lbl_names:
                old_lbl_val = lbl_names[key_name]
                if key_name not in NAME_LABEL_MAP:
                    continue
                new_lbl_val = NAME_LABEL_MAP[key_name]
                lbl_tmp[lbl == old_lbl_val] = new_lbl_val
            lbl_names_tmp = {}
            for key_name in lbl_names:
                if key_name not in NAME_LABEL_MAP:
                    continue
                lbl_names_tmp[key_name] = NAME_LABEL_MAP[key_name]

            # Assign the new label to lbl and lbl_names dict
            lbl = np.array(lbl_tmp, dtype=np.int8)
            b = np.max(lbl)
            c = np.max(lbl_tmp)
            lbl_names = lbl_names_tmp

            captions = ['%d: %s' % (l, name) for l, name in enumerate(lbl_names)]
            lbl_viz = utils.draw.draw_label(lbl, img, captions)
            out_dir = osp.basename(list[i]).replace('.', '_')
            out_dir = osp.join(osp.dirname(list[i]), out_dir)
            out_dir = json_file + 'label/'
            if not osp.exists(out_dir):
                os.mkdir(out_dir)

            PIL.Image.fromarray(img).save(osp.join(out_dir, '{}.png'.format(filename)))
            PIL.Image.fromarray(lbl).save(osp.join(out_dir, '{}_gt.png'.format(filename)))
            PIL.Image.fromarray(lbl_viz).save(osp.join(out_dir, '{}_viz.png'.format(filename)))

            with open(osp.join(out_dir, 'label_names.txt'), 'w') as f:
                for lbl_name in lbl_names:
                    f.write(lbl_name + '\n')

            warnings.warn('info.yaml is being replaced by label_names.txt')
            info = dict(label_names=lbl_names)
            with open(osp.join(out_dir, 'info.yaml'), 'w') as f:
                yaml.safe_dump(info, f, default_flow_style=False)

            print('Saved to: %s' % out_dir)


if __name__ == '__main__':
    main()