import torch
import os
import numpy as np
import cv2
from PIL import Image
from collections import defaultdict
from tqdm import tqdm
import mcubes
import open3d as o3d
from plyfile import PlyData, PlyElement
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.optim
import torch.distributed
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing
import os
from collections import OrderedDict
from ddp_model import NerfNetWithAutoExpo
import time
from data_loader_split import load_data_split
import numpy as np
from tensorboardX import SummaryWriter
from utils import img2mse, mse2psnr, img_HWC2CHW, colorize, TINY_NUMBER, colorize_np
import logging
import json
import os
import torch

# os.environ['CUDA_VISIBLE_DEVICES'] = '5,6'
logger = logging.getLogger(__package__)


def setup_logger():
    # create logger
    logger = logging.getLogger(__package__)
    # logger.setLevel(logging.DEBUG)
    logger.setLevel(logging.INFO)

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # create formatter
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')

    # add formatter to ch
    ch.setFormatter(formatter)

    # add ch to logger
    logger.addHandler(ch)


def intersect_sphere(ray_o, ray_d):
    '''
    ray_o, ray_d: [..., 3]
    compute the depth of the intersection point between this ray and unit sphere
    '''
    # note: d1 becomes negative if this mid point is behind camera
    d1 = -torch.sum(ray_d * ray_o, dim=-1) / torch.sum(ray_d * ray_d, dim=-1)
    p = ray_o + d1.unsqueeze(-1) * ray_d
    # consider the case where the ray does not intersect the sphere
    ray_d_cos = 1. / torch.norm(ray_d, dim=-1)
    p_norm_sq = torch.sum(p * p, dim=-1)
    if (p_norm_sq >= 1.).any():
        raise Exception(
            'Not all your cameras are bounded by the unit sphere; please make sure the cameras are normalized properly!')
    d2 = torch.sqrt(1. - p_norm_sq) * ray_d_cos

    return d1 + d2


def perturb_samples(z_vals):
    # get intervals between samples
    mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
    upper = torch.cat([mids, z_vals[..., -1:]], dim=-1)
    lower = torch.cat([z_vals[..., 0:1], mids], dim=-1)
    # uniform samples in those intervals
    t_rand = torch.rand_like(z_vals)
    z_vals = lower + (upper - lower) * t_rand  # [N_rays, N_samples]

    return z_vals


def sample_pdf(bins, weights, N_samples, det=False):
    '''
    :param bins: tensor of shape [..., M+1], M is the number of bins
    :param weights: tensor of shape [..., M]
    :param N_samples: number of samples along each ray
    :param det: if True, will perform deterministic sampling
    :return: [..., N_samples]
    '''
    # Get pdf
    weights = weights + TINY_NUMBER  # prevent nans
    pdf = weights / torch.sum(weights, dim=-1, keepdim=True)  # [..., M]
    cdf = torch.cumsum(pdf, dim=-1)  # [..., M]
    cdf = torch.cat([torch.zeros_like(cdf[..., 0:1]), cdf], dim=-1)  # [..., M+1]

    # Take uniform samples
    dots_sh = list(weights.shape[:-1])
    M = weights.shape[-1]

    min_cdf = 0.00
    max_cdf = 1.00  # prevent outlier samples

    if det:
        u = torch.linspace(min_cdf, max_cdf, N_samples, device=bins.device)
        u = u.view([1] * len(dots_sh) + [N_samples]).expand(dots_sh + [N_samples, ])  # [..., N_samples]
    else:
        sh = dots_sh + [N_samples]
        u = torch.rand(*sh, device=bins.device) * (max_cdf - min_cdf) + min_cdf  # [..., N_samples]

    # Invert CDF
    # [..., N_samples, 1] >= [..., 1, M] ----> [..., N_samples, M] ----> [..., N_samples,]
    above_inds = torch.sum(u.unsqueeze(-1) >= cdf[..., :M].unsqueeze(-2), dim=-1).long()

    # random sample inside each bin
    below_inds = torch.clamp(above_inds - 1, min=0)
    inds_g = torch.stack((below_inds, above_inds), dim=-1)  # [..., N_samples, 2]

    cdf = cdf.unsqueeze(-2).expand(dots_sh + [N_samples, M + 1])  # [..., N_samples, M+1]
    cdf_g = torch.gather(input=cdf, dim=-1, index=inds_g)  # [..., N_samples, 2]

    bins = bins.unsqueeze(-2).expand(dots_sh + [N_samples, M + 1])  # [..., N_samples, M+1]
    bins_g = torch.gather(input=bins, dim=-1, index=inds_g)  # [..., N_samples, 2]

    # fix numeric issue
    denom = cdf_g[..., 1] - cdf_g[..., 0]  # [..., N_samples]
    denom = torch.where(denom < TINY_NUMBER, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom

    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0] + TINY_NUMBER)

    return samples


def render_single_image(rank, world_size, models, ray_sampler, chunk_size):
    ##### parallel rendering of a single image
    ray_batch = ray_sampler.get_all()

    if (ray_batch['ray_d'].shape[0] // world_size) * world_size != ray_batch['ray_d'].shape[0]:
        raise Exception(
            'Number of pixels in the image is not divisible by the number of GPUs!\n\t# pixels: {}\n\t# GPUs: {}'.format(
                ray_batch['ray_d'].shape[0],
                world_size))

    # split into ranks; make sure different processes don't overlap
    rank_split_sizes = [ray_batch['ray_d'].shape[0] // world_size, ] * world_size
    rank_split_sizes[-1] = ray_batch['ray_d'].shape[0] - sum(rank_split_sizes[:-1])
    for key in ray_batch:
        if torch.is_tensor(ray_batch[key]):
            ray_batch[key] = torch.split(ray_batch[key], rank_split_sizes)[rank].to(rank)

    # split into chunks and render inside each process
    ray_batch_split = OrderedDict()
    for key in ray_batch:
        if torch.is_tensor(ray_batch[key]):
            ray_batch_split[key] = torch.split(ray_batch[key], chunk_size)

    # forward and backward
    ret_merge_chunk = [OrderedDict() for _ in range(models['cascade_level'])]
    for s in range(len(ray_batch_split['ray_d'])):
        ray_o = ray_batch_split['ray_o'][s]
        ray_d = ray_batch_split['ray_d'][s]
        min_depth = ray_batch_split['min_depth'][s]

        dots_sh = list(ray_d.shape[:-1])
        for m in range(models['cascade_level']):
            net = models['net_{}'.format(m)]
            # sample depths
            N_samples = models['cascade_samples'][m]
            if m == 0:
                # foreground depth
                fg_far_depth = intersect_sphere(ray_o, ray_d)  # [...,]
                fg_near_depth = min_depth  # [..., ]
                step = (fg_far_depth - fg_near_depth) / (N_samples - 1)
                fg_depth = torch.stack([fg_near_depth + i * step for i in range(N_samples)], dim=-1)  # [..., N_samples]

                # background depth
                bg_depth = torch.linspace(0., 1., N_samples).view(
                    [1, ] * len(dots_sh) + [N_samples, ]).expand(dots_sh + [N_samples, ]).to(rank)

                # delete unused memory
                del fg_near_depth
                del step
                torch.cuda.empty_cache()
            else:
                # sample pdf and concat with earlier samples
                fg_weights = ret['fg_weights'].clone().detach()
                fg_depth_mid = .5 * (fg_depth[..., 1:] + fg_depth[..., :-1])  # [..., N_samples-1]
                fg_weights = fg_weights[..., 1:-1]  # [..., N_samples-2]
                fg_depth_samples = sample_pdf(bins=fg_depth_mid, weights=fg_weights,
                                              N_samples=N_samples, det=True)  # [..., N_samples]
                fg_depth, _ = torch.sort(torch.cat((fg_depth, fg_depth_samples), dim=-1))

                # sample pdf and concat with earlier samples
                bg_weights = ret['bg_weights'].clone().detach()
                bg_depth_mid = .5 * (bg_depth[..., 1:] + bg_depth[..., :-1])
                bg_weights = bg_weights[..., 1:-1]  # [..., N_samples-2]
                bg_depth_samples = sample_pdf(bins=bg_depth_mid, weights=bg_weights,
                                              N_samples=N_samples, det=True)  # [..., N_samples]
                bg_depth, _ = torch.sort(torch.cat((bg_depth, bg_depth_samples), dim=-1))

                # delete unused memory
                del fg_weights
                del fg_depth_mid
                del fg_depth_samples
                del bg_weights
                del bg_depth_mid
                del bg_depth_samples
                torch.cuda.empty_cache()

            with torch.no_grad():
                ret = net(ray_o, ray_d, fg_far_depth, fg_depth, bg_depth)

            for key in ret:
                if key not in ['fg_weights', 'bg_weights']:
                    if torch.is_tensor(ret[key]):
                        if key not in ret_merge_chunk[m]:
                            ret_merge_chunk[m][key] = [ret[key].cpu(), ]
                        else:
                            ret_merge_chunk[m][key].append(ret[key].cpu())

                        ret[key] = None

            # clean unused memory
            torch.cuda.empty_cache()

    # merge results from different chunks
    for m in range(len(ret_merge_chunk)):
        for key in ret_merge_chunk[m]:
            ret_merge_chunk[m][key] = torch.cat(ret_merge_chunk[m][key], dim=0)

    # merge results from different processes
    if rank == 0:
        ret_merge_rank = [OrderedDict() for _ in range(len(ret_merge_chunk))]
        for m in range(len(ret_merge_chunk)):
            for key in ret_merge_chunk[m]:
                # generate tensors to store results from other processes
                sh = list(ret_merge_chunk[m][key].shape[1:])
                ret_merge_rank[m][key] = [torch.zeros(*[size, ] + sh, dtype=torch.float32) for size in rank_split_sizes]
                torch.distributed.gather(ret_merge_chunk[m][key], ret_merge_rank[m][key])
                ret_merge_rank[m][key] = torch.cat(ret_merge_rank[m][key], dim=0).reshape(
                    (ray_sampler.H, ray_sampler.W, -1)).squeeze()
                # print(m, key, ret_merge_rank[m][key].shape)
    else:  # send results to main process
        for m in range(len(ret_merge_chunk)):
            for key in ret_merge_chunk[m]:
                torch.distributed.gather(ret_merge_chunk[m][key])

    # only rank 0 program returns
    if rank == 0:
        return ret_merge_rank
    else:
        return None


def label_img_to_color(img):
    label_to_color = {
        0: [0, 0, 0],
        1: [0, 0, 0],
        2: [0, 0, 0],
        3: [0, 0, 0],
        4: [0, 0, 0],
        5: [111, 74, 0],
        6: [81, 0, 81],
        7: [128, 64, 128],
        8: [244, 35, 232],
        9: [250, 170, 160],
        10: [230, 150, 140],
        11: [70, 70, 70],
        12: [102, 102, 156],
        13: [190, 153, 153],
        14: [180, 165, 180],
        15: [150, 100, 100],
        16: [150, 120, 90],
        17: [153, 153, 153],
        18: [153, 153, 153],
        19: [250, 170, 30],
        20: [220, 220, 0],
        21: [107, 142, 35],
        22: [152, 251, 152],
        23: [70, 130, 180],
        24: [220, 20, 60],
        25: [255, 0, 0],
        26: [0, 0, 142],
        27: [0, 0, 70],
        28: [0, 60, 100],
        29: [0, 0, 90],
        30: [0, 0, 110],
        31: [0, 80, 100],
        32: [0, 0, 230],
        33: [119, 11, 32],
        34: [64, 128, 128],
        35: [190, 153, 153],
        36: [150, 120, 90],
        37: [153, 153, 153],
        38: [0, 64, 64],
        39: [0, 128, 192],
        40: [128, 64, 0],
        41: [64, 64, 128],
        42: [102, 0, 0],
        43: [51, 0, 51],
        44: [32, 32, 32],
        45: [32, 196, 32],
        46: [32, 32, 196]
    }

    img_height, img_width = img.shape

    img_color = np.zeros((img_height, img_width, 3))
    for row in range(img_height):
        for col in range(img_width):
            label = img[row, col]

            img_color[row, col] = np.array(label_to_color[label])

    return img_color


def log_view_to_tb(writer, global_step, log_data, gt_img, gt_semantic, gt_depth, mask, prefix=''):
    rgb_im = img_HWC2CHW(torch.from_numpy(gt_img))
    writer.add_image(prefix + 'rgb_gt', rgb_im, global_step)

    # logits_2_label = lambda x: torch.argmax(torch.nn.functional.softmax(x, dim=-1), dim=-1)
    logits_2_label = lambda x: torch.argmax(x, dim=-1)

    semantic_im = torch.from_numpy(label_img_to_color(gt_semantic))
    semantic_im = img_HWC2CHW(semantic_im)
    writer.add_image(prefix + 'semantic_gt', semantic_im, global_step)

    gt_depth[gt_depth > 200] = 200
    depth_im = img_HWC2CHW(torch.from_numpy(colorize_np(gt_depth, cmap_name='jet', append_cbar=True,
                                                        mask=mask)))
    writer.add_image(prefix + 'depth_gt', depth_im, global_step)

    for m in range(len(log_data)):
        rgb_im = img_HWC2CHW(log_data[m]['rgb'])
        rgb_im = torch.clamp(rgb_im, min=0., max=1.)  # just in case diffuse+specular>1
        writer.add_image(prefix + 'level_{}/rgb'.format(m), rgb_im, global_step)

        rgb_im = img_HWC2CHW(log_data[m]['fg_rgb'])
        rgb_im = torch.clamp(rgb_im, min=0., max=1.)  # just in case diffuse+specular>1
        writer.add_image(prefix + 'level_{}/fg_rgb'.format(m), rgb_im, global_step)
        depth = log_data[m]['fg_depth']
        depth_im = img_HWC2CHW(colorize(depth, cmap_name='jet', append_cbar=True,
                                        mask=mask))
        writer.add_image(prefix + 'level_{}/fg_depth'.format(m), depth_im, global_step)

        rgb_im = img_HWC2CHW(log_data[m]['bg_rgb'])
        rgb_im = torch.clamp(rgb_im, min=0., max=1.)  # just in case diffuse+specular>1
        writer.add_image(prefix + 'level_{}/bg_rgb'.format(m), rgb_im, global_step)

        semantic_im = logits_2_label(log_data[m]['semantic'])
        semantic_im = label_img_to_color(semantic_im.numpy())
        # semantic_im[gt_semantic == 0, :] = 0
        semantic_im = torch.from_numpy(semantic_im)
        semantic_im = img_HWC2CHW(semantic_im)
        writer.add_image(prefix + 'level_{}/semantic'.format(m), semantic_im, global_step)

        semantic_im = logits_2_label(log_data[m]['fg_semantic'])
        semantic_im = label_img_to_color(semantic_im.numpy())
        # semantic_im[gt_semantic == 0, :] = 0
        semantic_im = torch.from_numpy(semantic_im)
        semantic_im = img_HWC2CHW(semantic_im)
        writer.add_image(prefix + 'level_{}/fg_semantic'.format(m), semantic_im, global_step)

        semantic_im = logits_2_label(log_data[m]['bg_semantic'])
        semantic_im = label_img_to_color(semantic_im.numpy())
        # semantic_im[gt_semantic == 0, :] = 0
        semantic_im = torch.from_numpy(semantic_im)
        semantic_im = img_HWC2CHW(semantic_im)
        writer.add_image(prefix + 'level_{}/bg_semantic'.format(m), semantic_im, global_step)

        depth = log_data[m]['bg_depth']
        depth_im = img_HWC2CHW(colorize(depth, cmap_name='jet', append_cbar=True,
                                        mask=mask))
        writer.add_image(prefix + 'level_{}/bg_depth'.format(m), depth_im, global_step)
        bg_lambda = log_data[m]['bg_lambda']
        bg_lambda_im = img_HWC2CHW(colorize(bg_lambda, cmap_name='hot', append_cbar=True,
                                            mask=mask))
        writer.add_image(prefix + 'level_{}/bg_lambda'.format(m), bg_lambda_im, global_step)

        depth = log_data[m]['depth_fgbg']
        depth_im = img_HWC2CHW(colorize(depth, cmap_name='jet', append_cbar=True,
                                        mask=mask))
        writer.add_image(prefix + 'level_{}/depth_fgbg'.format(m), depth_im, global_step)


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    # port = np.random.randint(12355, 12399)
    # os.environ['MASTER_PORT'] = '{}'.format(port)
    os.environ['MASTER_PORT'] = '12356'
    # initialize the process group
    torch.distributed.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    torch.distributed.destroy_process_group()


def create_nerf(rank, args):
    ###### create network and wrap in ddp; each process should do this
    # fix random seed just to make sure the network is initialized with same weights at different processes
    torch.manual_seed(777)
    # very important!!! otherwise it might introduce extra memory in rank=0 gpu
    torch.cuda.set_device(rank)

    models = OrderedDict()
    models['cascade_level'] = args.cascade_level
    models['cascade_samples'] = [int(x.strip()) for x in args.cascade_samples.split(',')]
    for m in range(models['cascade_level']):
        img_names = None
        if args.optim_autoexpo:
            # load training image names for autoexposure
            f = os.path.join(args.basedir, args.expname, 'train_images.json')
            with open(f) as file:
                img_names = json.load(file)
        net = NerfNetWithAutoExpo(args, optim_autoexpo=args.optim_autoexpo, img_names=img_names).to(rank)
        net = DDP(net, device_ids=[rank], output_device=rank, find_unused_parameters=True)
        # net = DDP(net, device_ids=[rank], output_device=rank)
        optim = torch.optim.Adam(net.parameters(), lr=args.lrate)
        models['net_{}'.format(m)] = net
        models['optim_{}'.format(m)] = optim

    start = -1

    ###### load pretrained weights; each process should do this
    if (args.ckpt_path is not None) and (os.path.isfile(args.ckpt_path)):
        ckpts = [args.ckpt_path]
    else:
        ckpts = [os.path.join(args.basedir, args.expname, f)
                 for f in sorted(os.listdir(os.path.join(args.basedir, args.expname))) if f.endswith('.pth')]

    def path2iter(path):
        tmp = os.path.basename(path)[:-4]
        idx = tmp.rfind('_')
        return int(tmp[idx + 1:])

    ckpts = sorted(ckpts, key=path2iter)
    logger.info('Found ckpts: {}'.format(ckpts))
    if len(ckpts) > 0 and not args.no_reload:
        fpath = ckpts[-1]
        logger.info('Reloading from: {}'.format(fpath))
        start = path2iter(fpath)
        # configure map_location properly for different processes
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        to_load = torch.load(fpath, map_location=map_location)
        for m in range(models['cascade_level']):
            for name in ['net_{}'.format(m), 'optim_{}'.format(m)]:
                models[name].load_state_dict(to_load[name])

    return start, models


def ddp_train_nerf(rank, args):
    ###### set up multi-processing
    setup(rank, args.world_size)
    ###### set up logger
    logger = logging.getLogger(__package__)
    setup_logger()

    ###### decide chunk size according to gpu memory
    logger.info('gpu_mem: {}'.format(torch.cuda.get_device_properties(rank).total_memory))
    if torch.cuda.get_device_properties(rank).total_memory / 1e9 > 14:
        logger.info('setting batch size according to 24G gpu')
        args.N_rand = 1024
        args.chunk_size = 8192
    else:
        logger.info('setting batch size according to 12G gpu before N_rand: {} chunk_size: {}'.format(args.N_rand,
                                                                                                      args.chunk_size))
        args.N_rand = 1024
        args.chunk_size = 8192
        logger.info('setting batch size according to 12G gpu after N_rand: {} chunk_size: {}'.format(args.N_rand,
                                                                                                     args.chunk_size))

    ###### Create log dir and copy the config file
    if rank == 0:
        os.makedirs(os.path.join(args.basedir, args.expname), exist_ok=True)
        f = os.path.join(args.basedir, args.expname, 'args.txt')
        with open(f, 'w') as file:
            for arg in sorted(vars(args)):
                attr = getattr(args, arg)
                file.write('{} = {}\n'.format(arg, attr))
        if args.config is not None:
            f = os.path.join(args.basedir, args.expname, 'config.txt')
            with open(f, 'w') as file:
                file.write(open(args.config, 'r').read())
    torch.distributed.barrier()

    start, models = create_nerf(rank, args)

    time0 = time.time()
    # randomly sample rays and move to device

    N = args.N_grid
    xmin, xmax = args.x_range
    ymin, ymax = args.y_range
    zmin, zmax = args.z_range
    # assert xmax-xmin == ymax-ymin == zmax-zmin, 'the ranges must have the same length!'
    x = np.linspace(xmin, xmax, N)
    y = np.linspace(ymin, ymax, N)
    z = np.linspace(zmin, zmax, N)

    xyz_ = torch.FloatTensor(np.stack(np.meshgrid(x, y, z), -1).reshape(-1, 3)).cuda()
    dir_ = torch.zeros_like(xyz_).cuda()

    # predict sigma (occupancy) for each grid location
    print('Predicting occupancy ...')


    for m in range(models['cascade_level']):
        net = models['net_{}'.format(m)]
        if m == 1:
            with torch.no_grad():
                B = xyz_.shape[0]
                out_chunks = []
                for i in tqdm(range(0, B, args.chunk_size)):
                    out_chunks += [net(xyz_[i:i+args.chunk_size])]
                rgbsigma = torch.cat(out_chunks, 0)

            sigma = rgbsigma.cpu().numpy()
            sigma = np.maximum(sigma, 0).reshape(N, N, N)

            # perform marching cube algorithm to retrieve vertices and triangle mesh
            print('Extracting mesh ...')
            vertices, triangles = mcubes.marching_cubes(sigma, args.sigma_threshold)

            ##### Until mesh extraction here, it is the same as the original repo. ######

            vertices_ = (vertices / N).astype(np.float32)
            ## invert x and y coordinates (WHY? maybe because of the marching cubes algo)
            x_ = (ymax - ymin) * vertices_[:, 1] + ymin
            y_ = (xmax - xmin) * vertices_[:, 0] + xmin
            vertices_[:, 0] = x_
            vertices_[:, 1] = y_
            vertices_[:, 2] = (zmax - zmin) * vertices_[:, 2] + zmin
            vertices_.dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]

            face = np.empty(len(triangles), dtype=[('vertex_indices', 'i4', (3,))])
            face['vertex_indices'] = triangles

            PlyData([PlyElement.describe(vertices_[:, 0], 'vertex'),
                     PlyElement.describe(face, 'face')]).write(f'{args.scene}.ply')

            # remove noise in the mesh by keeping only the biggest cluster
            print('Removing noise ...')
            mesh = o3d.io.read_triangle_mesh(f"{args.scene}.ply")
            idxs, count, _ = mesh.cluster_connected_triangles()
            max_cluster_idx = np.argmax(count)
            triangles_to_remove = [i for i in range(len(face)) if idxs[i] != max_cluster_idx]
            mesh.remove_triangles_by_index(triangles_to_remove)
            mesh.remove_unreferenced_vertices()
            print(f'Mesh has {len(mesh.vertices) / 1e6:.2f} M vertices and {len(mesh.triangles) / 1e6:.2f} M faces.')

            vertices_ = np.asarray(mesh.vertices).astype(np.float32)
            triangles = np.asarray(mesh.triangles)

            # perform color prediction
            # Step 0. define constants (image width, height and intrinsics)

            K = np.array([[552.554261, 0.000000, 682.049453],
                    [0.000000, 552.554261, 238.769549],
                    [0.000000, 0.000000, 1.000000]]).astype(np.float32)

            ray_samplers = load_data_split(args.datadir, args.scene, split='train',
                                           try_load_min_depth=args.load_min_depth)

            # Step 1. transform vertices into world coordinate
            N_vertices = len(vertices_)
            vertices_homo = np.concatenate([vertices_, np.ones((N_vertices, 1))], 1)  # (N, 4)
            vertices_homo[:, :3] *= 48.932
            vertices_homo[:, :3] -= np.array([2643.5, -1026.1, -116.5])

            ## buffers to store the final averaged color
            non_occluded_sum = np.zeros((N_vertices, 1))
            v_color_sum = np.zeros((N_vertices, 3))

            # Step 2. project the vertices onto each training image to infer the color
            print('Fusing colors ...')
            for idx in tqdm(range(len(ray_samplers))):
                ## read image of this pose
                image = ray_samplers[idx].get_img()
                H, W = image.shape[:2]

                ## read the camera to world relative pose
                P_c2w = ray_samplers[idx].c2w_mat
                P_c2w[:3, 3] *= 48.932
                P_c2w[:3, 3] -= np.array((2643.5, -1026.1, -116.5))
                P_w2c = np.linalg.inv(P_c2w)[:3]  # (3, 4)
                ## project vertices from world coordinate to camera coordinate
                vertices_cam = (P_w2c @ vertices_homo.T)  # (3, N)
                ## project vertices from camera coordinate to pixel coordinate
                vertices_image = (K @ vertices_cam).T  # (N, 3)
                depth = vertices_image[:, -1:] + 1e-5  # the depth of the vertices, used as far plane
                vertices_image = vertices_image[:, :2] / depth
                vertices_image = vertices_image.astype(np.float32)
                vertices_image[:, 0] = np.clip(vertices_image[:, 0], 0, W - 1)
                vertices_image[:, 1] = np.clip(vertices_image[:, 1], 0, H - 1)

                ## compute the color on these projected pixel coordinates
                ## using bilinear interpolation.
                ## NOTE: opencv's implementation has a size limit of 32768 pixels per side,
                ## so we split the input into chunks.
                colors = []
                remap_chunk = int(3e4)
                for i in range(0, N_vertices, remap_chunk):
                    colors += [cv2.remap(image,
                                         vertices_image[i:i + remap_chunk, 0],
                                         vertices_image[i:i + remap_chunk, 1],
                                         interpolation=cv2.INTER_LINEAR)[:, 0]]
                colors = np.vstack(colors)  # (N_vertices, 3)
                non_occluded = np.ones_like(non_occluded_sum)

                v_color_sum += colors * non_occluded
                non_occluded_sum += non_occluded

            # Step 3. combine the output and write to file
            v_colors = v_color_sum/non_occluded_sum * 255.0

            v_colors = v_colors.astype(np.uint8)
            v_colors.dtype = [('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
            vertices_.dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
            vertex_all = np.empty(N_vertices, vertices_.dtype.descr + v_colors.dtype.descr)
            for prop in vertices_.dtype.names:
                vertex_all[prop] = vertices_[prop][:, 0]
            for prop in v_colors.dtype.names:
                vertex_all[prop] = v_colors[prop][:, 0]

            face = np.empty(len(triangles), dtype=[('vertex_indices', 'i4', (3,))])
            face['vertex_indices'] = triangles

            PlyData([PlyElement.describe(vertex_all, 'vertex'),
                     PlyElement.describe(face, 'face')]).write(f'{args.scene}.ply')

            print('Done!')

    ### end of core optimization loop
    dt = time.time() - time0

    # clean up for multi-processing
    cleanup()


def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, help='config file path')
    parser.add_argument("--expname", type=str, default='nuscenes', help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/', help='where to store ckpts and logs')
    # dataset options
    parser.add_argument("--datadir", type=str, default='./data', help='input data directory')
    parser.add_argument("--scene", type=str, default='road', help='scene name')
    parser.add_argument("--testskip", type=int, default=8,
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')
    # model size
    parser.add_argument("--netdepth", type=int, default=4, help='layers in coarse network')
    parser.add_argument("--netwidth", type=int, default=64, help='channels per layer in coarse network')
    parser.add_argument("--use_viewdirs", action='store_true', help='use full 5D input instead of 3D')
    # checkpoints
    parser.add_argument("--no_reload", action='store_true', help='do not reload weights from saved ckpt')
    parser.add_argument("--ckpt_path", type=str, default=None,
                        help='specific weights npy file to reload for coarse network')
    # batch size
    parser.add_argument("--N_rand", type=int, default=32,
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--chunk_size", type=int, default=16,
                        help='number of rays processed in parallel, decrease if running out of memory')
    # iterations
    parser.add_argument("--N_iters", type=int, default=250001,
                        help='number of iterations')
    # render only
    parser.add_argument("--render_splits", type=str, default='test',
                        help='splits to render')
    # cascade training
    parser.add_argument("--cascade_level", type=int, default=1,
                        help='number of cascade levels')
    parser.add_argument("--cascade_samples", type=str, default='64,64',
                        help='samples at each level')
    # multiprocess learning
    parser.add_argument("--world_size", type=int, default='-1',
                        help='number of processes')
    # optimize autoexposure
    parser.add_argument("--optim_autoexpo", action='store_true',
                        help='optimize autoexposure parameters')
    parser.add_argument("--lambda_autoexpo", type=float, default=1., help='regularization weight for autoexposure')

    # learning rate options
    parser.add_argument("--lrate", type=float, default=5e-4, help='learning rate')
    parser.add_argument("--lrate_decay_factor", type=float, default=0.1,
                        help='decay learning rate by a factor every specified number of steps')
    parser.add_argument("--lrate_decay_steps", type=int, default=5000,
                        help='decay learning rate by a factor every specified number of steps')
    # rendering options
    parser.add_argument("--det", action='store_true', help='deterministic sampling for coarse and fine samples')
    parser.add_argument("--max_freq_log2", type=int, default=10,
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--max_freq_log2_viewdirs", type=int, default=4,
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--load_min_depth", action='store_true', help='whether to load min depth')
    # logging/saving options
    parser.add_argument("--i_print", type=int, default=100, help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img", type=int, default=2, help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=10000, help='frequency of weight ckpt saving')
    parser.add_argument("--semantic_wgt", type=float, default=4e-2, help='semantic_wgt')
    parser.add_argument('--N_grid', type=int, default=256,
                        help='size of the grid on 1 side, larger=higher resolution')
    parser.add_argument('--x_range', nargs="+", type=float, default=[-0.8, 0.8],
                        help='x range of the object')
    parser.add_argument('--y_range', nargs="+", type=float, default=[-0.8, 0.8],
                        help='y range of the object')
    parser.add_argument('--z_range', nargs="+", type=float, default=[-0.8, 0.8],
                        help='z range of the object')
    parser.add_argument('--sigma_threshold', type=float, default=1,
                        help='threshold to consider a location is occupied')
    parser.add_argument('--occ_threshold', type=float, default=0.5,
                        help='''threshold to consider a vertex is occluded.
                                    larger=fewer occluded pixels''')
    parser.add_argument('--near_t', type=float, default=1.0,
                        help='the near bound factor to start the ray')

    return parser


def train():
    parser = config_parser()
    args = parser.parse_args()
    logger.info(parser.format_values())

    if args.world_size == -1:
        args.world_size = torch.cuda.device_count()
        logger.info('Using # gpus: {}'.format(args.world_size))
    torch.multiprocessing.spawn(ddp_train_nerf,
                                args=(args,),
                                nprocs=args.world_size,
                                join=True)


if __name__ == '__main__':
    setup_logger()
    train()


