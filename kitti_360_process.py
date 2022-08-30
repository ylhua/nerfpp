import json
import cv2
import numpy as np
import os
import glob

save_image_path = '/media/huayanling/d9020a15-be70-448c-a6c3-9400b7f1a855/data/kitti_360/seglane_15/'
ori_image_path = '/media/huayanling/d9020a15-be70-448c-a6c3-9400b7f1a855/data/kitti_360/download_2d_perspective/KITTI-360/data_2d_raw/2013_05_28_drive_0010_sync/image_00/data_rect/'
ori_semantic_path = '/media/huayanling/d9020a15-be70-448c-a6c3-9400b7f1a855/data/kitti_360/static_scene/all/semantic_with_lane/semantic'
ori_semantic_vis_path = '/media/huayanling/d9020a15-be70-448c-a6c3-9400b7f1a855/data/kitti_360/static_scene/all/semantic_with_lane/semantic_vis/'
transient_mask_vis_path = '/media/huayanling/d9020a15-be70-448c-a6c3-9400b7f1a855/data/kitti_360/seglane_15/all/transient_mask_vis/'
transient_mask_path = '/media/huayanling/d9020a15-be70-448c-a6c3-9400b7f1a855/data/kitti_360/seglane_15/all/transient_mask/'

frames_all_toge = []
frames_all_no_sensor_train = []
frames_all_no_sensor_test = []

i = 0
intrincs = np.array([[552.554261, 0.000000, 682.049453],
                    [0.000000, 552.554261, 238.769549],
                    [0.000000, 0.000000, 1.000000]])

def find_files(dir, exts):
    if os.path.isdir(dir):
        # types should be ['*.png', '*.jpg']
        files_grabbed = []
        for ext in exts:
            files_grabbed.extend(glob.glob(os.path.join(dir, ext)))
        if len(files_grabbed) > 0:
            files_grabbed = sorted(files_grabbed)
        return files_grabbed
    else:
        return []

def parse_txt(filename):
    assert os.path.isfile(filename)
    nums = open(filename).read().split()
    return np.array([float(x) for x in nums]).reshape([3026, 17]).astype(np.float32)

semantic_file_list = find_files(ori_semantic_path, exts=['*.png'])
poses = parse_txt('/media/huayanling/d9020a15-be70-448c-a6c3-9400b7f1a855/data/kitti_360/data_poses/2013_05_28_drive_0010_sync/cam0_to_world.txt')
pose_name  = []
for pose_i in range(len(poses)):
    pose_name.append(str(np.array(poses[pose_i,  0]).astype(np.int)).zfill(10) + '.png')

i = 0
for cur_frame_num in range(0, len(semantic_file_list)):
    cur_image_name = semantic_file_list[cur_frame_num].split('/')[-1]
    rgb_image = cv2.imread(ori_image_path + cur_image_name)
    semantic_vis_im = cv2.imread(ori_semantic_vis_path + cur_image_name)
    is_exit = cur_image_name not in pose_name
    if cur_image_name not in pose_name:
        continue
    semantic = cv2.imread(semantic_file_list[cur_frame_num], cv2.IMREAD_UNCHANGED)
    # semantic_tmp = np.zeros_like(rgb_image)
    # semantic_tmp[:,:,0] = semantic
    # semantic_tmp[:, :, 1] = semantic
    # semantic_tmp[:, :, 2] = semantic
    a = np.max(semantic)
    b = np.min(semantic)
    # pereson = np.where(semantic == 24)
    transient_mask_vis = np.zeros_like(semantic)
    transient_mask_vis[np.where((semantic <= 34))] = 255
    transient_mask_vis[np.where((semantic < 24))] = 0
    cv2.imwrite(transient_mask_vis_path + semantic_file_list[cur_frame_num].split('/')[-1], transient_mask_vis)

    transient_mask = np.zeros_like(semantic)
    transient_mask[np.where((semantic <= 34))] = 1
    transient_mask[np.where((semantic < 24))] = 0
    cv2.imwrite(transient_mask_path + semantic_file_list[cur_frame_num].split('/')[-1], transient_mask)

    # semantic[np.where(transient_mask == 1)] = 45
    # semantic[np.where(semantic < 0)] = 45
    # semantic[np.where(semantic > 44)] = 45

    rgb_image = cv2.imread(ori_image_path + cur_image_name)
    cv2.imwrite(os.path.join(save_image_path, 'all/rgb', cur_image_name), rgb_image)

    cv2.imwrite(os.path.join(save_image_path, 'all/semantic', cur_image_name), semantic)

    cur_frame_sensor_info_four = {}
    cur_frame_sensor_info_four.update({'camera_intrinsic': intrincs.tolist()})

    pose_index = pose_name.index(cur_image_name)
    transforms_matrics = poses[pose_index, :][1:].reshape((4,4))
    transforms_matrics[0:3, 3] = transforms_matrics[0:3, 3]
    cur_frame_sensor_info_four.update({'transform_matrix': transforms_matrics.tolist()})

    cur_frame_sensor_info_four.update({'file_name': cur_image_name})

    frames_all_toge.append(cur_frame_sensor_info_four)

    if i % 4 == 0:
        frames_all_no_sensor_test.append(cur_frame_sensor_info_four)
        image_save_path = save_image_path + 'test/rgb/' + cur_image_name
        cv2.imwrite(image_save_path, rgb_image)

        semantic_save_path = save_image_path + 'test/semantic/' + cur_image_name
        cv2.imwrite(semantic_save_path, semantic)

        transient_mask_save_path = save_image_path + 'test/transient_mask/' + cur_image_name
        cv2.imwrite(transient_mask_save_path, transient_mask)

        transient_mask_save_path = save_image_path + 'test/transient_mask_vis/' + cur_image_name
        cv2.imwrite(transient_mask_save_path, transient_mask_vis)

        image_save_path = save_image_path + 'test/semantic_rgb/' + cur_image_name
        cv2.imwrite(image_save_path, semantic_vis_im)
    else:
        frames_all_no_sensor_train.append(cur_frame_sensor_info_four)
        image_save_path = save_image_path + 'train/rgb/' + cur_image_name
        cv2.imwrite(image_save_path, rgb_image)

        semantic_save_path = save_image_path + 'train/semantic/' + cur_image_name
        cv2.imwrite(semantic_save_path, semantic)

        transient_mask_save_path = save_image_path + 'train/transient_mask/' + cur_image_name
        cv2.imwrite(transient_mask_save_path, transient_mask)

        transient_mask_save_path = save_image_path + 'train/transient_mask_vis/' + cur_image_name
        cv2.imwrite(transient_mask_save_path, transient_mask_vis)

        image_save_path = save_image_path + 'train/semantic_rgb/' + cur_image_name
        cv2.imwrite(image_save_path, semantic_vis_im)

    i = i + 1
    if i > 50:
        break


with open(save_image_path + 'config/' + 'all_no_sensor_test.json', "a") as dump_f:
    json.dump(frames_all_no_sensor_test, dump_f, indent=4, ensure_ascii=False)

with open(save_image_path + 'config/' + 'all_no_sensor_train.json', "a") as dump_f:
    json.dump(frames_all_no_sensor_train, dump_f, indent=4, ensure_ascii=False)

with open(save_image_path + 'config/' + 'all_no_sensor_all.json', "a") as dump_f:
    json.dump(frames_all_toge, dump_f, indent=4, ensure_ascii=False)


