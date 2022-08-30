#%matplotlib inline
#https://blog.csdn.net/weixin_44398263/article/details/120063785
from nuscenes.nuscenes import NuScenes
import json
from pyquaternion import Quaternion
import cv2
import numpy as np

#nusc = NuScenes(version='v1.0-trainval', dataroot='/media/linger/新加卷1/IE_Downloads/v1.0-trainval_meta/v1.0-trainval_meta', verbose=True)
save_image_path = '/media/linger/ubuntu/null_max/label_manual/nerfplusplus/nuscenes/road/'
ori_image_path = '/media/linger/新加卷/IE_Downloads/v1.0-trainval01_keyframes/'
ori_semantic_path = '/media/linger/ubuntu/null_max/label_manual/barf/nuscenes/all/semantic/'
transient_mask_path = '/media/linger/ubuntu/null_max/label_manual/nerfplusplus/nuscenes/all/transient_mask/'
#nusc = NuScenes(version='v1.0-mini', dataroot='/media/linger/ubuntu/null_max/code/v1.0-mini', verbose=True)
nusc = NuScenes(version='v1.0-trainval', dataroot='/media/linger/新加卷/IE_Downloads/v1.0-trainval_meta/v1.0-trainval_meta', verbose=True)
#nusc.render_egoposes_on_map(log_location='singapore-onenorth')
nusc.list_scenes()
#my_scene_auto = nusc.scene[0]
my_scene_auto = nusc.get('scene', '91c071bcc1ad4fa1b555399e1cfbab79')
cur_frame_token = my_scene_auto['first_sample_token']
sensor_list = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
frames_all = {'CAM_FRONT':[], 'CAM_FRONT_LEFT':[], 'CAM_FRONT_RIGHT':[], 'CAM_BACK':[], 'CAM_BACK_LEFT':[], 'CAM_BACK_RIGHT':[]}
frames_all_toge = []
frames_all_no_sensor_train = []
frames_all_no_sensor_test = []
frames_all_no_sensor_all = []
i = 0

for cur_frame_num in range(my_scene_auto['nbr_samples']):
    cur_frame_sample = nusc.get('sample', cur_frame_token)
    cur_frame_token = cur_frame_sample['next']
    cur_franme_info = {'CAM_FRONT': [], 'CAM_FRONT_LEFT': [], 'CAM_FRONT_RIGHT': [], 'CAM_BACK': [], 'CAM_BACK_LEFT': [],
                  'CAM_BACK_RIGHT': []}
    for cur_sensor in sensor_list:
        cur_frame_sensor_data = cur_frame_sample['data'][cur_sensor]
        cur_frame_sensor_data = nusc.get('sample_data', cur_frame_sensor_data)
        cur_frame_sensor_info = {}
        cur_frame_sensor_info.update({'file_name': cur_frame_sensor_data['filename']})

        ego_pose = nusc.get('ego_pose', cur_frame_sensor_data['ego_pose_token'])
        ego_pose_rotation = Quaternion(ego_pose['rotation']).rotation_matrix.tolist()
        cur_frame_sensor_info.update(
            {'ego_pose': {'translation': ego_pose['translation'], 'rotation': ego_pose_rotation}})

        calib_info = nusc.get('calibrated_sensor', cur_frame_sensor_data['calibrated_sensor_token'])
        calib_rotation = Quaternion(calib_info['rotation']).rotation_matrix.tolist()
        cur_frame_sensor_info.update(
            {'calib_info': {'translation': calib_info['translation'], 'rotation': calib_rotation}})

        cur_frame_sensor_info.update(
            {'camera_intrinsic': calib_info['camera_intrinsic']})

        cur_franme_info[cur_sensor].append(cur_frame_sensor_info)
        frames_all[cur_sensor].append(cur_frame_sensor_info)
        cur_image = cv2.imread(ori_image_path + cur_frame_sensor_data['filename'])
        image_save_path = save_image_path + 'all/samples/' + cur_frame_sensor_data['filename'].split('/')[-1]
        # cv2.imwrite(image_save_path, cur_image)

        cur_frame_sensor_info_four = {}
        egopose_matrix = np.vstack([
            np.hstack((ego_pose_rotation,
                       np.array(ego_pose['translation'])[:, None])),
            np.array([0, 0, 0, 1])
        ])

        calib_matrix = np.vstack([
            np.hstack((calib_rotation,
                       np.array(calib_info['translation'])[:, None])),
            np.array([0, 0, 0, 1])
        ])

        cur_frame_sensor_info_four.update({'file_name': cur_frame_sensor_data['filename']})
        cur_frame_sensor_info_four.update({'calib_info': calib_matrix.tolist()})
        cur_frame_sensor_info_four.update({'ego_pose': egopose_matrix.tolist()})
        cur_frame_sensor_info_four.update({'camera_intrinsic': calib_info['camera_intrinsic']})
        cur_frame_sensor_info_four.update({'img_size': cur_image.shape})

        frames_all_no_sensor_all.append(cur_frame_sensor_info_four)

        if i % 6 == 0:
            frames_all_no_sensor_test.append(cur_frame_sensor_info_four)
            image_save_path = save_image_path + 'test/samples/' + cur_frame_sensor_data['filename'].split('/')[-1]
            cv2.imwrite(image_save_path, cur_image)

            # semantic_label = cv2.imread(ori_semantic_path + cur_frame_sensor_data['filename'].split('/')[-1].split('.')[0] + '_gt.png')
            # a = np.max(semantic_label)
            # semantic_save_path = save_image_path + 'test/semantic/' + cur_frame_sensor_data['filename'].split('/')[-1].split('.')[0] + '.png'
            # cv2.imwrite(semantic_save_path, semantic_label)

            # transient_mask = cv2.imread(
            #     transient_mask_path + cur_frame_sensor_data['filename'].split('/')[-1].split('.')[0] + '_gt.png')
            # a = np.max(transient_mask)
            # transient_mask_save_path = save_image_path + 'test/transient_mask/' + \
            #                      cur_frame_sensor_data['filename'].split('/')[-1].split('.')[0] + '.png'
            # cv2.imwrite(transient_mask_save_path, transient_mask)
        else:
            frames_all_no_sensor_train.append(cur_frame_sensor_info_four)
            image_save_path = save_image_path + 'train/samples/' + cur_frame_sensor_data['filename'].split('/')[-1]
            cv2.imwrite(image_save_path, cur_image)

            # semantic_label = cv2.imread(
            #     ori_semantic_path + cur_frame_sensor_data['filename'].split('/')[-1].split('.')[0] + '_gt.png')
            # a = np.max(semantic_label)
            # semantic_save_path = save_image_path + 'train/semantic/' + \
            #                      cur_frame_sensor_data['filename'].split('/')[-1].split('.')[0] + '.png'
            # cv2.imwrite(semantic_save_path, semantic_label)

            # transient_mask = cv2.imread(
            #     transient_mask_path + cur_frame_sensor_data['filename'].split('/')[-1].split('.')[0] + '_gt.png')
            # a = np.max(transient_mask)
            # transient_mask_save_path = save_image_path + 'train/transient_mask/' + \
            #                            cur_frame_sensor_data['filename'].split('/')[-1].split('.')[0] + '.png'
            # cv2.imwrite(transient_mask_save_path, transient_mask)

    frames_all_toge.append(cur_franme_info)
    i = i + 1

for cur_sensor in sensor_list:
    with open(save_image_path + 'config/' + str(cur_sensor) + ".json", "a") as dump_f:
            json.dump(frames_all[cur_sensor], dump_f, indent=4, ensure_ascii=False)

with open(save_image_path + 'config/' + 'all_no_sensor_test.json', "a") as dump_f:
    json.dump(frames_all_no_sensor_test, dump_f, indent=4, ensure_ascii=False)

with open(save_image_path + 'config/' + 'all_no_sensor_train.json', "a") as dump_f:
    json.dump(frames_all_no_sensor_train, dump_f, indent=4, ensure_ascii=False)

with open(save_image_path + 'config/' + 'all_no_sensor_all.json', "a") as dump_f:
    json.dump(frames_all_no_sensor_all, dump_f, indent=4, ensure_ascii=False)

with open(save_image_path + 'config/' + "all.json", "a") as dump_f:
    json.dump(frames_all_toge, dump_f, indent=4, ensure_ascii=False)


