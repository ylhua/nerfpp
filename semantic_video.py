import cv2
import os
import numpy as np
semantic_file = '/home/huayanling/projects/null_max/code/tmp/logs/seg_lane/render_test_252000_ori/semantic_gt/'
img_file = '/home/huayanling/projects/null_max/code/tmp/logs/seg_lane/render_test_252000_ori/rgb/'
blend_save_file = '/home/huayanling/projects/null_max/code/tmp/logs/seg_lane/render_test_252000_ori/semantic_blend/'
gt_semantic = '/home/huayanling/projects/null_max/code/tmp/logs/seg_lane/render_test_252000_ori/semantic/'

list = os.listdir(semantic_file)
for i in range(0, len(list)):
    path = os.path.join(semantic_file, list[i])

    label_vis_img = cv2.imread(path)
    img_name = list[i].split('_')[-1]
    img_ori_dir = os.path.join(img_file, img_name)
    gt_ori_dir = os.path.join(gt_semantic, 'semantic_' + img_name)
    img = cv2.imread(img_ori_dir)
    # img = np.concatenate((img, np.zeros_like(img[:, :80, :])), axis=1)
    # label_vis_img = np.concatenate((label_vis_img, np.ones_like(label_vis_img[:, :13, :]) * 250.0), axis=1)
    gt_semantic_img = cv2.imread(gt_ori_dir)
    img_ori = np.concatenate([img, label_vis_img, gt_semantic_img], axis=0)
    save_name = img_name.zfill(10)
    cv2.imwrite(os.path.join(blend_save_file, save_name), img_ori)

fsp = 4
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_path = '/home/huayanling/projects/null_max/code/tmp/logs/seg_lane/render_test_252000_ori/test_semantic.avi'  # 改 ①
video_out = cv2.VideoWriter(video_path, fourcc, fsp, (1408, 1128))  # 改 ③

list = sorted(os.listdir(blend_save_file))
for i in range(1, len(list)):
    frame = cv2.imread(blend_save_file + list[i])  # 改 ⑤
    video_out.write(frame)

semantic_file = 'F:/null_max/expriment/nerf_road/semantic/'
img_file = 'F:\\null_max\\expriment\\nerf_road\\rgb/'
save_file = 'F:/null_max/label_manual/nerfplusplus/nuscenes/test/seg_vis/'
blend_save_file = 'F:/null_max/expriment/nerf_road/blend_vis/'

list = os.listdir(semantic_file)
for i in range(0, len(list)):
    path = os.path.join(semantic_file, list[i])

    label_vis_img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img_ori_dir = os.path.join(img_file, list[i].split('.')[0][9:] + '.jpg')
    img_ori = np.zeros_like(label_vis_img)
    img = cv2.imread(img_ori_dir, cv2.IMREAD_GRAYSCALE)
    img_ori[:, :, 0] = img
    img_ori[:, :, 1] = img
    img_ori[:, :, 2] = img
    img_vis = img_ori * 0.5 + label_vis_img * 0.5

    cv2.imwrite(os.path.join(save_file, list[i]), label_vis_img)
    cv2.imwrite(os.path.join(blend_save_file, list[i]), img_vis)

fsp = 2
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_path = 'F:/null_max/expriment/nerf_road/11.avi'  # 改 ①
video_out = cv2.VideoWriter(video_path, fourcc, fsp, (1600, 900))  # 改 ③

list = os.listdir(blend_save_file)
for i in range(0, len(list)):
    frame = cv2.imread(blend_save_file + list[i])  # 改 ⑤
    video_out.write(frame)
