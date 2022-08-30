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
    radius = diagonal * 1.5

    translate = -center
    scale = target_radius / radius

    return translate, scale


def normalize_cam_dict_nuscenes(in_cam_dict_file, out_cam_dict_file, another_cam_dict_file, target_radius=1., in_geometry_file=None, out_geometry_file=None):
    with open(in_cam_dict_file) as fp:
        in_cam_dict = json.load(fp)

    with open(another_cam_dict_file) as fp:
        another_cam_dict = json.load(fp)

    translate, scale = get_tf_cams_nuscenes(in_cam_dict, another_cam_dict, target_radius=target_radius)

    # if in_geometry_file is not None and out_geometry_file is not None:
    #     # check this page if you encounter issue in file io: http://www.open3d.org/docs/0.9.0/tutorial/Basic/file_io.html
    #     geometry = o3d.io.read_triangle_mesh(in_geometry_file)
    #
    #     tf_translate = np.eye(4)
    #     tf_translate[:3, 3:4] = translate
    #     tf_scale = np.eye(4)
    #     tf_scale[:3, :3] *= scale
    #     tf = np.matmul(tf_scale, tf_translate)
    #
    #     geometry_norm = geometry.transform(tf)
    #     o3d.io.write_triangle_mesh(out_geometry_file, geometry_norm)
  
    def transform_pose(C2W, translate, scale):
        cam_center = C2W[:3, 3]
        cam_center = (cam_center + translate) * scale
        C2W[:3, 3] = cam_center
        return C2W

    out_cam_dict = copy.deepcopy(in_cam_dict)
    out_all = []
    for img_name in out_cam_dict:
        C2W = np.array(img_name['transform_matrix']).reshape((4, 4))
        C2W = transform_pose(C2W, translate, scale)
        assert(np.isclose(np.linalg.det(C2W[:3, :3]), 1.))

        img_name.update({'transform_matrix': C2W.tolist()})
        out_all.append(img_name)

        out_C2W_dir = out_cam_dict_file + 'test/poses/' + img_name['file_name'].split('/')[-1].split('.')[0] + '.txt'
        c2w = str(C2W.flatten()).replace('[', '').replace(']', '')
        c2w = c2w.replace("'", '').replace(',', '').replace('\n', '') + '\n'
        with open(out_C2W_dir, 'w') as f:
            f.write(c2w)

        cam_intri = np.array(img_name['camera_intrinsic']).reshape((3, 3))
        cam_intri_four = np.zeros_like(C2W)
        cam_intri_four[:3, :3] = cam_intri
        cam_intri_four[3, 3] = 1
        img_name.update({'K': list(cam_intri_four.flatten())})
        img_name.update({'K': cam_intri_four.tolist()})

        image_size =  np.array([376, 1408, 3])
        img_name.update({'img_size': list(image_size.tolist())})
        out_intrinsics_dir = out_cam_dict_file + 'test/intrinsics/' + img_name['file_name'].split('/')[-1].split('.')[0] + '.txt'

        cam_intri_four_s = str(cam_intri_four.flatten()).replace('[', '').replace(']', '')
        cam_intri_four_s = cam_intri_four_s.replace("'", '').replace(',', '').replace('\n', '') + '\n'
        with open(out_intrinsics_dir, 'w') as f:
            f.write(str(cam_intri_four_s))

    with open(out_cam_dict_file + 'transforms_test.json', 'w') as fp:
        json.dump(out_all, fp, indent=2, sort_keys=True)


if __name__ == '__main__':
    in_cam_dict_file = '/media/huayanling/d9020a15-be70-448c-a6c3-9400b7f1a855/data/kitti_360/seglane_15/config/all_no_sensor_test.json'
    anaother_cam_dict_file = '/media/huayanling/d9020a15-be70-448c-a6c3-9400b7f1a855/data/kitti_360/seglane_15/config/all_no_sensor_train.json'
    out_cam_dict_file = '/media/huayanling/d9020a15-be70-448c-a6c3-9400b7f1a855/data/kitti_360/seglane_15/'
    normalize_cam_dict_nuscenes(in_cam_dict_file, out_cam_dict_file, anaother_cam_dict_file, target_radius=1.)
