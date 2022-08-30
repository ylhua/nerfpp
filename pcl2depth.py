#!/usr/bin/python
# -*- coding: utf-8 -*-


#################
## Import modules
#################
import sys
# walk directories
import glob
# access to OS functionality
import os
# call processes
import subprocess
# copy things
import copy
# numpy
import numpy as np
# open3d
import open3d
# from lineset import LineMesh
# matplotlib for colormaps
import matplotlib.cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# scipy
from scipy import interpolate
# struct for reading binary ply files
import struct
# parse arguments
import argparse

try:
    import matplotlib.colors
    from PIL import PILLOW_VERSION
    from PIL import Image
except:
    pass

HUGE_NUMBER = 1e10
TINY_NUMBER = 1e-6      # float32 only has 7 decimal digits precision
to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)
########################################################################################################################
#
########################################################################################################################
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import matplotlib as mpl
from matplotlib import cm
import cv2
import imageio

def get_vertical_colorbar(h, vmin, vmax, cmap_name='jet', label=None):
    fig = Figure(figsize=(2, 12), dpi=100)
    fig.subplots_adjust(right=1.5)
    canvas = FigureCanvasAgg(fig)

    # Do some plotting.
    ax = fig.add_subplot(111)
    cmap = cm.get_cmap(cmap_name)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    tick_cnt = 6
    tick_loc = np.linspace(vmin, vmax, tick_cnt)
    cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                    norm=norm,
                                    ticks=tick_loc,
                                    orientation='vertical')

    tick_label = ['{:3.2f}'.format(x) for x in tick_loc]
    cb1.set_ticklabels(tick_label)

    cb1.ax.tick_params(labelsize=18, rotation=0)

    if label is not None:
        cb1.set_label(label)

    fig.tight_layout()

    canvas.draw()
    s, (width, height) = canvas.print_to_buffer()

    im = np.frombuffer(s, np.uint8).reshape((height, width, 4))

    im = im[:, :, :3].astype(np.float32) / 255.
    if h != im.shape[0]:
        w = int(im.shape[1] / im.shape[0] * h)
        im = cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)

    return im


def colorize_np(y, cmap_name='jet', mask=None, append_cbar=False):
    x = y.copy()
    x[x > 200] = 200
    if mask is not None:
        # vmin, vmax = np.percentile(x[mask], (1, 99))
        vmin = np.min(x[mask])
        vmax = np.max(x[mask])
        vmin = vmin - np.abs(vmin) * 0.01
        x[np.logical_not(mask)] = vmin
        x = np.clip(x, vmin, vmax)
        # print(vmin, vmax)
    else:
        vmin = x.min()
        vmax = x.max() + TINY_NUMBER

    x = (x - vmin) / (vmax - vmin)
    # x = np.clip(x, 0., 1.)

    cmap = cm.get_cmap(cmap_name)
    x_new = cmap(x)[:, :, :3]

    if mask is not None:
        mask = np.float32(mask[:, :, np.newaxis])
        x_new = x_new * mask + np.zeros_like(x_new) * (1. - mask)

    cbar = get_vertical_colorbar(h=x.shape[0], vmin=vmin, vmax=vmax, cmap_name=cmap_name)

    if append_cbar:
        x_new = np.concatenate((x_new, np.zeros_like(x_new[:, :5, :]), cbar), axis=1)
        return x_new
    else:
        return x_new, cbar

#################
## Helper classes
#################
os.environ['KITTI360_DATASET'] = '/media/huayanling/d9020a15-be70-448c-a6c3-9400b7f1a855/data/kitti_360'

# annotation helper
from kitti360scripts.helpers.annotation import Annotation3D, Annotation3DPly, global2local
from kitti360scripts.helpers.project import Camera
from kitti360scripts.helpers.labels import name2label, id2label, kittiId2label
from kitti360scripts.helpers.ply import read_ply

def parse_txt(filename):
    assert os.path.isfile(filename)
    nums = open(filename).read().split()
    return np.array([float(x) for x in nums]).reshape([3026, 17]).astype(np.float32)

# the main class that parse fused point clouds
class Kitti360Viewer3D(object):

    # Constructor
    def __init__(self, seq=0, showStatic=True):

        # The sequence of the image we currently working on
        self.currentSequence = ""
        # Image extension
        self.imageExt = ".png"
        # Filenames of all images in current city
        self.images = []
        self.imagesCityFull = []
        # Ground truth type
        self.gtType = 'semantic'
        # Add contour to semantic map
        self.semanticCt = True
        # Add contour to instance map
        self.instanceCt = True
        # The object that is highlighted and its label. An object instance
        self.highlightObj = None
        self.highlightObjSparse = None
        self.highlightObjLabel = None
        # The current object the mouse points to. It's index in self.labels
        self.mouseObj = -1
        # The current object the mouse points to. It's index in self.labels
        self.mousePressObj = -1
        self.mouseSemanticId = -1
        self.mouseInstanceId = -1
        # show camera or not
        self.showCamera = False
        self.downSampleEvery = -1
        # show bbox wireframe or mesh
        self.showWireframe = False
        self.show3DInstanceOnly = True
        # show static or dynamic point clouds
        self.showStatic = showStatic
        # show visible point clouds only
        self.showVisibleOnly = False
        # colormap for instances
        self.cmap = matplotlib.cm.get_cmap('Set1')
        self.cmap_length = 9
        # colormap for confidence
        self.cmap_conf = matplotlib.cm.get_cmap('plasma')

        if 'KITTI360_DATASET' in os.environ:
            kitti360Path = os.environ['KITTI360_DATASET']
        else:
            kitti360Path = os.path.join(os.path.dirname(
                os.path.realpath(__file__)), '..', '..')

        sequence = '2013_05_28_drive_%04d_sync' % seq
        self.label3DPcdPath = os.path.join(kitti360Path, 'data_3d_semantics')
        self.label3DBboxPath = os.path.join(kitti360Path, 'data_3d_bboxes')
        # self.annotation3D = Annotation3D(self.label3DBboxPath, sequence)
        self.annotation3DPly = Annotation3DPly(self.label3DPcdPath, sequence)
        self.sequence = sequence

        self.pointClouds = {}
        self.Rz = np.eye(3)
        self.bboxes = []
        self.bboxes_window = []
        self.accumuData = []

    def assignColorConfidence(self, confidence):
        color = self.cmap_conf(confidence)[:, :3]
        return color

    def assignColorDynamic(self, timestamps):
        color = np.zeros((timestamps.size, 3))
        for uid in np.unique(timestamps):
            color[timestamps == uid] = self.getColor(uid)
        return color

    def getLabelFilename(self, currentFile):
        # Generate the filename of the label file
        filename = os.path.basename(currentFile)
        search = [lb for lb in self.label_images if filename in lb]
        if not search:
            return ""
        filename = os.path.normpath(search[0])
        return filename

    def assignColor(self, globalIds, gtType='semantic'):
        if not isinstance(globalIds, (np.ndarray, np.generic)):
            globalIds = np.array(globalIds)[None]
        color = np.zeros((globalIds.size, 3))
        for uid in np.unique(globalIds):
            semanticId, instanceId = global2local(uid)
            if gtType=='semantic':
                color[globalIds==uid] = id2label[semanticId].color
            elif instanceId>0:
                color[globalIds==uid] = self.getColor(instanceId)
            else:
                color[globalIds==uid] = (96,96,96) # stuff objects in instance mode
        color = color.astype(np.float)/255.0
        return color

    def assignColorConfidence(self, confidence):
        color = self.cmap_conf(confidence)[:,:3]
        return color

    def loadWindow(self, pcdFile, colorType='semantic', isLabeled=True, isDynamic=False):
        window = pcdFile.split(os.sep)[-2]

        print('Loading %s ' % pcdFile)

        # load ply data using open3d for visualization
        if window in self.pointClouds.keys():
            pcd = self.pointClouds[window]
        else:
            # pcd = open3d.io.read_point_cloud(pcdFile)
            data = read_ply(pcdFile)
            points = np.vstack((data['x'], data['y'], data['z'])).T
            color = np.vstack((data['red'], data['green'], data['blue'])).T
            pcd = open3d.geometry.PointCloud()
            pcd.points = open3d.utility.Vector3dVector(points)
            pcd.colors = open3d.utility.Vector3dVector(color.astype(np.float) / 255.)

        # assign color
        if colorType == 'semantic' or colorType == 'instance':
            globalIds = data['instance']
            ptsColor = self.assignColor(globalIds, colorType)
            pcd.colors = open3d.utility.Vector3dVector(ptsColor)
        elif colorType == 'bbox':
            ptsColor = np.asarray(pcd.colors)
            pcd.colors = open3d.utility.Vector3dVector(ptsColor)
        elif colorType == 'confidence':
            confidence = data[:, -1]
            ptsColor = self.assignColorConfidence(confidence)
            pcd.colors = open3d.utility.Vector3dVector(ptsColor)
        elif colorType != 'rgb':
            raise ValueError("Color type can only be 'rgb', 'bbox', 'semantic', 'instance'!")

        if self.showVisibleOnly:
            isVisible = data['visible']
            pcd = pcd.select_by_index(np.where(isVisible)[0])

        if self.downSampleEvery > 1:
            print(np.asarray(pcd.points).shape)
            pcd = pcd.uniform_down_sample(self.downSampleEvery)
            print(np.asarray(pcd.points).shape)

        return pcd, points

def pcl2depth(pcd):
    depth_save_path = '/media/huayanling/d9020a15-be70-448c-a6c3-9400b7f1a855/data/kitti_360/static_scene/depth'
    cam0_pose = '/media/huayanling/d9020a15-be70-448c-a6c3-9400b7f1a855/data/kitti_360/data_poses/2013_05_28_drive_0010_sync/cam0_to_world.txt'
    image_size = (376, 1408)
    intrincs = np.array([[552.554261, 0.000000, 682.049453],
                              [0.000000, 552.554261, 238.769549],
                              [0.000000, 0.000000, 1.000000]])
    poses = parse_txt(cam0_pose)
    pcd = pcd.T
    pcd = np.vstack((pcd, np.ones((1,pcd.shape[1]))))
    pose_names = []
    for pose_i in range(325, 384):
        pose_name = str(np.array(poses[pose_i, 0]).astype(np.int)).zfill(10) + '.png'
        pose_names.append(pose_name)
        transforms_matrics = poses[pose_i, :][1:].reshape((4, 4))
        w2c = np.linalg.inv(transforms_matrics)
        pcd_c = np.dot(w2c, pcd)
        EPS = 1.0e-16
        valid = pcd_c[2, :] > EPS
        z = pcd_c[2, valid]
        u = np.round(pcd_c[0, valid] * intrincs[0,0] / z + intrincs[0,2]).astype(int)
        v = np.round(pcd_c[1, valid] * intrincs[1,1] / z + intrincs[1,2]).astype(int)

        valid = np.bitwise_and(np.bitwise_and((u >= 0), (u < image_size[1])),
                               np.bitwise_and((v >= 0), (v < image_size[0])))
        u, v, z = u[valid], v[valid], z[valid]

        img_z = np.full((image_size[0], image_size[1]), np.inf)
        for ui, vi, zi in zip(u, v, z):
            img_z[vi, ui] = min(img_z[vi, ui], zi)

        img_z_shift = np.array([img_z, \
                                np.roll(img_z, 1, axis=0), \
                                np.roll(img_z, -1, axis=0), \
                                np.roll(img_z, 1, axis=1), \
                                np.roll(img_z, -1, axis=1)])
        img_z = np.min(img_z_shift, axis=0)

        img_z_shift = np.array([img_z, \
                                np.roll(img_z, 1, axis=0), \
                                np.roll(img_z, -1, axis=0), \
                                np.roll(img_z, 1, axis=1), \
                                np.roll(img_z, -1, axis=1)])

        img_z = np.min(img_z_shift, axis=0)

        img_z[img_z == np.inf] = 0

        img_z = impletesky(pose_name, img_z)

        im = colorize_np(img_z, cmap_name='jet', append_cbar=True)
        im = to8b(im)
        imageio.imwrite(os.path.join(depth_save_path, 'depth_vis', pose_name), im)

        depth_ori_name = pose_name.split('.')[0] + '.csv'
        np.savetxt(os.path.join(depth_save_path, 'depth_data', depth_ori_name), img_z, fmt='%.12f', delimiter=',', newline='\n')

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

def impletesky(pose_name, depth_map):
    ori_semantic_path = '/media/huayanling/d9020a15-be70-448c-a6c3-9400b7f1a855/data/kitti_360/static_scene/all/semantic'
    semantic_file_list = find_files(ori_semantic_path, exts=['*.png'])
    pose_names = []
    for pose_i in range(0, len(semantic_file_list)):
        pose_names.append(semantic_file_list[pose_i].split('/')[-1])
    if pose_name in pose_names:
        semantic_label = cv2.imread(os.path.join(ori_semantic_path,  pose_name))
        sky_pos = semantic_label[:, :, 0] == 23
        depth_map[sky_pos] = 200.0
    return depth_map

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--sequence', type=int, default=0,
                        help='The sequence to visualize')
    parser.add_argument('--mode', choices=['rgb', 'semantic', 'instance', 'confidence', 'bbox'], default='semantic',
                        help='The modality to visualize')
    parser.add_argument('--max_bbox', type=int, default=100,
                        help='The maximum number of bounding boxes to visualize')

    args = parser.parse_args()

    v = Kitti360Viewer3D(args.sequence)

    pcdFileList = v.annotation3DPly.pcdFileList
    points_all = []
    for idx, pcdFile in enumerate(pcdFileList):
        pcd, points = v.loadWindow(pcdFile, args.mode)
        if len(np.asarray(pcd.points)) == 0:
            print('Warning: skipping empty point cloud!')
            continue
        points_all.append(points)
    points_all = np.concatenate(points_all)
    pcl2depth(points_all[:537000, :])
    exit()

