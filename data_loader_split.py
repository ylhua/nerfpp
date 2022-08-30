import os
import numpy as np
import imageio
import logging
from nerf_sample_ray_split import RaySamplerSingleImage
import glob

logger = logging.getLogger(__package__)


########################################################################################################################
# camera coordinate system: x-->right, y-->down, z-->scene (opencv/colmap convention)
# poses is camera-to-world
########################################################################################################################
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


def load_data_split(basedir, scene, split, skip=1, try_load_min_depth=True, only_img_files=False):
    def parse_txt(filename):
        assert os.path.isfile(filename)
        nums = open(filename).read().split()
        return np.array([float(x) for x in nums]).reshape([4, 4]).astype(np.float32)

    if basedir[-1] == '/':  # remove trailing '/'
        basedir = basedir[:-1]

    split_dir = '{}/{}/{}'.format(basedir, scene, split)

    if only_img_files:
        img_files = find_files('{}/rgb'.format(split_dir), exts=['*.png', '*.jpg'])
        return img_files

    # camera parameters files
    intrinsics_files = find_files('{}/intrinsics'.format(split_dir), exts=['*.txt'])
    pose_files = find_files('{}/poses'.format(split_dir), exts=['*.txt'])
    logger.info('raw intrinsics_files: {}'.format(len(intrinsics_files)))
    logger.info('raw pose_files: {}'.format(len(pose_files)))

    intrinsics_files = intrinsics_files[::skip]
    pose_files = pose_files[::skip]
    cam_cnt = len(pose_files)

    # img files
    img_files = find_files('{}/rgb'.format(split_dir), exts=['*.png', '*.jpg'])
    if len(img_files) > 0:
        logger.info('raw img_files: {}'.format(len(img_files)))
        img_files = img_files[::skip]
        assert (len(img_files) == cam_cnt)
    else:
        img_files = [None, ] * cam_cnt

    # img files
    transient_mask_file = find_files('{}/transient_mask'.format(split_dir), exts=['*.png', '*.jpg'])
    if len(transient_mask_file) > 0:
        logger.info('raw img_files: {}'.format(len(transient_mask_file)))
        transient_mask_file = transient_mask_file[::skip]
        assert (len(transient_mask_file) == cam_cnt)
    else:
        transient_mask_file = [None, ] * cam_cnt

    # semantic files
    semantic_files = find_files('{}/semantic'.format(split_dir), exts=['*.png', '*.jpg'])
    if len(semantic_files) > 0:
        logger.info('raw img_files: {}'.format(len(semantic_files)))
        semantic_files = semantic_files[::skip]
        assert (len(semantic_files) == cam_cnt)
    else:
        semantic_files = [None, ] * cam_cnt

    # depth files
    depth_gt_files = find_files('{}/depth_sky'.format(split_dir), exts=['*.csv'])
    if len(depth_gt_files) > 0:
        logger.info('raw img_files: {}'.format(len(depth_gt_files)))
        depth_gt_files = depth_gt_files[::skip]
        assert (len(depth_gt_files) == cam_cnt)
    else:
        depth_gt_files = [None, ] * cam_cnt

    # mask files
    mask_files = find_files('{}/mask'.format(split_dir), exts=['*.png', '*.jpg'])
    if len(mask_files) > 0:
        logger.info('raw mask_files: {}'.format(len(mask_files)))
        mask_files = mask_files[::skip]
        assert (len(mask_files) == cam_cnt)
    else:
        mask_files = [None, ] * cam_cnt

    # min depth files
    mindepth_files = find_files('{}/min_depth'.format(split_dir), exts=['*.png', '*.jpg'])
    if try_load_min_depth and len(mindepth_files) > 0:
        logger.info('raw mindepth_files: {}'.format(len(mindepth_files)))
        mindepth_files = mindepth_files[::skip]
        assert (len(mindepth_files) == cam_cnt)
    else:
        mindepth_files = [None, ] * cam_cnt

    # assume all images have the same size as training image
    train_imgfile = find_files('{}/{}/train/rgb'.format(basedir, scene), exts=['*.png', '*.jpg'])[0]
    train_im = imageio.imread(train_imgfile)
    H, W = train_im.shape[:2]

    # create ray samplers
    ray_samplers = []
    for i in range(cam_cnt):
        intrinsics = parse_txt(intrinsics_files[i])
        pose = parse_txt(pose_files[i])

        # pose[:3, :3] = np.array(((0, -1, 0), (1, 0, 0), (0, 0, 1))) @ np.array(((1, 0, 0), (0, 0, 1), (0, -1, 0))) @ pose[:3, :3]
        # pose[2, 3] = pose[2, 3] + 0.4

        # read max depth
        try:
            max_depth = float(open('{}/max_depth.txt'.format(split_dir)).readline().strip())
        except:
            max_depth = None

        ray_samplers.append(RaySamplerSingleImage(H=H, W=W, intrinsics=intrinsics, c2w=pose,
                                                  img_path=img_files[i],
                                                  semantic_path=semantic_files[i],
                                                  mask_path=mask_files[i],
                                                  min_depth_path=mindepth_files[i],
                                                  max_depth=max_depth,
                                                  transient_mask=transient_mask_file[i],
                                                  pose_path=pose_files[i],
                                                  depth_gt_path=depth_gt_files[i])
                            )

    logger.info('Split {}, # views: {}'.format(split, cam_cnt))

    return ray_samplers
