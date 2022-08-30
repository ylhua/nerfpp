import imageio


# def main():
#     #### images path
#     root_path = '/home/huayanling/projects/null_max/expriment/nerfplus_ori/semantic/'
#     image_list = ['semantic_n008-2018-08-01-15-16-36-0400__CAM_BACK__1533151203537558.jpg', 'semantic_n008-2018-08-01-15-16-36-0400__CAM_BACK__1533151206437563.jpg',
#                   'semantic_n008-2018-08-01-15-16-36-0400__CAM_BACK__1533151209437558.jpg']
#     for i in range(len(image_list)):
#         image_list[i] = root_path + image_list[i]
#     # save name
#     gif_name = '/home/huayanling/projects/null_max/expriment/nerfplus_ori/Vivica_Fox.gif'
#     # duration between images
#     duration = 0.5
#
#     #### read images and write in gif
#     images = []
#     for image_name in image_list:
#         images.append(imageio.imread(image_name))
#     imageio.mimwrite(gif_name, images, 'GIF', duration=duration)
#
#     print('success')
#
#
# if __name__ == "__main__":
#     main()


import cv2
import os
import numpy as np
semantic_file = '/home/huayanling/projects/null_max/expriment/nerfplus_ori/semantic/'
img_file = '/home/huayanling/projects/null_max/expriment/nerfplus_ori/rgb/'
blend_save_file = '/home/huayanling/projects/null_max/expriment/nerfplus_ori/tt/'

list = os.listdir(semantic_file)
for i in range(0, len(list)):
    path = os.path.join(semantic_file, list[i])

    label_vis_img = cv2.imread(path)
    img_ori_dir = os.path.join(img_file, list[i])
    img = cv2.imread(path)
    img_ori = np.concatenate([label_vis_img, img], axis=1)
    cv2.imwrite(os.path.join(blend_save_file, list[i]), img_ori)


gif_name = '/home/huayanling/projects/null_max/expriment/nerfplus_ori/Vivica_Fox.gif'
img_path = "/home/huayanling/projects/null_max/expriment/nerfplus_ori/tt/"  # 改 ②
list = os.listdir(img_path)
duration = 0.5
images = []
for i in range(0, len(list)):
    frame = cv2.imread(img_path + list[i])  # 改 ⑤
    images.append(imageio.imread(img_path + list[i]))

imageio.mimwrite(gif_name, images, 'GIF', duration=duration)
print('success')
#
# semantic_file = 'F:/null_max/expriment/nerf_road/semantic/'
# img_file = 'F:\\null_max\\expriment\\nerf_road\\rgb/'
# save_file = 'F:/null_max/label_manual/nerfplusplus/nuscenes/test/seg_vis/'
# blend_save_file = 'F:/null_max/expriment/nerf_road/blend_vis/'
#
# list = os.listdir(semantic_file)
# for i in range(0, len(list)):
#     path = os.path.join(semantic_file, list[i])
#
#     label_vis_img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
#     img_ori_dir = os.path.join(img_file, list[i].split('.')[0][9:] + '.jpg')
#     img_ori = np.zeros_like(label_vis_img)
#     img = cv2.imread(img_ori_dir, cv2.IMREAD_GRAYSCALE)
#     img_ori[:, :, 0] = img
#     img_ori[:, :, 1] = img
#     img_ori[:, :, 2] = img
#     img_vis = img_ori * 0.5 + label_vis_img * 0.5
#
#     cv2.imwrite(os.path.join(save_file, list[i]), label_vis_img)
#     cv2.imwrite(os.path.join(blend_save_file, list[i]), img_vis)
#
# fsp = 2
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# video_path = 'F:/null_max/expriment/nerf_road/11.avi'  # 改 ①
# video_out = cv2.VideoWriter(video_path, fourcc, fsp, (1600, 900))  # 改 ③
#
# list = os.listdir(blend_save_file)
# for i in range(0, len(list)):
#     frame = cv2.imread(blend_save_file + list[i])  # 改 ⑤
#     video_out.write(frame)
