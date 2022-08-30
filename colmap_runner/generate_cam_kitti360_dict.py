import numpy as np
import json
import copy
import open3d as o3d



def get_tf_cams_nuscenes(cam_dict, another_dict, target_radius=1.):
    cam_centers = []
    for im_name in cam_dict:
        C2W = np.array(im_name['transform_matrix']).reshape((4, 4))
        cam_centers.append(C2W[:3, 3:4])

    for im_name in another_dict:
        C2W = np.array(im_name['transform_matrix']).reshape((4, 4))
        cam_centers.append(C2W[:3, 3:4])

    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center
    scale = target_radius / radius

    return translate, scale


def normalize_cam_dict_nuscenes(in_cam_dict_file, out_cam_dict_file, another_cam_dict_file, target_radius=1., in_geometry_file=None, out_geometry_file=None):
    with open(in_cam_dict_file) as fp:
        in_cam_dict = json.load(fp)

    with open(another_cam_dict_file) as fp:
        another_cam_dict = json.load(fp)

    translate, scale = get_tf_cams_nuscenes(in_cam_dict, another_cam_dict, target_radius=target_radius)
  
    def transform_pose(C2W, translate, scale):
        cam_center = C2W[:3, 3]
        cam_center = (cam_center + translate) * scale
        C2W[:3, 3] = cam_center
        return C2W

    out_cam_dict = copy.deepcopy(in_cam_dict)
    out_all = []
    start_pose = np.array([-2648.0, 991.0, 121.0])
    end_pose = np.array([-2637.97, 1090.25, 115.0])
    step = (end_pose - start_pose) / 300
    rotation_mat = np.array([
            [0.9836829900741577,-0.03583788126707077,0.1763100028038025],
            [-0.17910799384117126,-0.10232380032539368,0.9784939885139465],
            [-0.01702599972486496,-0.9941055774688721,-0.10707200318574905]
        ])

    cam_intri = np.array([[552.554261, 0.000000, 682.049453],
                          [0.000000, 552.554261, 238.769549],
                          [0.000000, 0.000000, 1.000000]])

    for idx in range(0, 300):
        pose_info = {}
        cur_pose = np.identity(4)
        cur_pose[:3, :3] = rotation_mat
        cur_pose[:3, 3] = start_pose + step * idx + [2648, -992, -112]
        C2W = transform_pose(cur_pose, translate, scale)
        assert(np.isclose(np.linalg.det(C2W[:3, :3]), 1.))

        pose_info.update({'transform_matrix': C2W.tolist()})

        out_C2W_dir = out_cam_dict_file + 'generate/poses/' + str(idx) + '.txt'
        c2w = str(C2W.flatten()).replace('[', '').replace(']', '')
        c2w = c2w.replace("'", '').replace(',', '').replace('\n', '') + '\n'
        with open(out_C2W_dir, 'w') as f:
            f.write(c2w)

        cam_intri_four = np.identity(4)
        cam_intri_four[:3, :3] = cam_intri
        pose_info.update({'K': list(cam_intri_four.flatten())})
        pose_info.update({'K': cam_intri_four.tolist()})

        image_size =  np.array([376, 1408, 3])
        pose_info.update({'img_size': list(image_size.tolist())})
        out_intrinsics_dir = out_cam_dict_file + 'generate/intrinsics/' + str(idx) + '.txt'

        cam_intri_four_s = str(cam_intri_four.flatten()).replace('[', '').replace(']', '')
        cam_intri_four_s = cam_intri_four_s.replace("'", '').replace(',', '').replace('\n', '') + '\n'
        with open(out_intrinsics_dir, 'w') as f:
            f.write(str(cam_intri_four_s))

        out_all.append(pose_info)

    with open(out_cam_dict_file + 'generate/generate_poses.json', 'w') as fp:
        json.dump(out_all, fp, indent=2, sort_keys=True)


if __name__ == '__main__':
    in_cam_dict_file = '/media/huayanling/d9020a15-be70-448c-a6c3-9400b7f1a855/data/kitti_360/static_scene/config/all_no_sensor_test.json'
    anaother_cam_dict_file = '/media/huayanling/d9020a15-be70-448c-a6c3-9400b7f1a855/data/kitti_360/static_scene/config/all_no_sensor_train.json'
    out_cam_dict_file = '/media/huayanling/d9020a15-be70-448c-a6c3-9400b7f1a855/data/kitti_360/static_scene/'
    normalize_cam_dict_nuscenes(in_cam_dict_file, out_cam_dict_file, anaother_cam_dict_file, target_radius=1.)
