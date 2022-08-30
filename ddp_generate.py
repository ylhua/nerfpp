import torch
# import torch.nn as nn
import torch.optim
import torch.distributed
# from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing
import numpy as np
import os
# from collections import OrderedDict
# from ddp_model import NerfNet
import time
from data_loader_split import load_data_split
from utils import mse2psnr, colorize_np, to8b
import imageio
from ddp_train_nerf import config_parser, setup_logger, setup, cleanup, render_single_image, create_nerf
import logging
from sklearn.metrics import confusion_matrix

logger = logging.getLogger(__package__)

def label_img_to_color(img):
    label_to_color = {
    0: [0, 0, 0],
    1: [0, 0, 0],
    2: [0, 0, 0],
    3: [0,  0,  0],
    4: [0,  0,  0],
    5: [111, 74,  0],
    6: [81,  0, 81],
    7: [128, 64,128],
    8: [244, 35,232],
    9: [250,170,160],
    10: [230,150,140],
    11: [70, 70, 70],
    12: [102,102,156],
    13: [190,153,153],
    14: [180,165,180],
    15: [150,100,100],
    16: [150,120, 90],
    17: [153,153,153],
    18: [153,153,153],
    19: [250,170, 30],
    20: [220,220,  0],
    21: [107,142, 35],
    22: [152,251,152],
    23: [70,130,180],
    24: [220, 20, 60],
    25: [255,  0,  0],
    26: [0,  0,142],
    27: [0,  0, 70],
    28: [0, 60,100],
    29: [0,  0, 90],
    30: [0,  0,110],
    31: [0, 80,100],
    32: [0,  0,230],
    33: [119, 11, 32],
    34: [64,128,128],
    35: [190,153,153],
    36: [150,120, 90],
    37: [153,153,153],
    38: [0,   64, 64],
    39: [0,  128,192],
    40: [128, 64,  0],
    41: [64,  64,128],
    42: [102,  0,  0],
    43: [51,  0, 51],
    44: [32, 32, 32],
    45: [0, 0, 0]
    }

    img_height, img_width = img.shape

    img_color = np.zeros((img_height, img_width, 3))
    for row in range(img_height):
        for col in range(img_width):
            label = img[row, col]

            img_color[row, col] = np.array(label_to_color[label])

    return img_color

def nanmean(data, **args):
    # This makes it ignore the first 'background' class
    return np.ma.masked_array(data, np.isnan(data)).mean(**args)
    # In np.ma.masked_array(data, np.isnan(data), elements of data == np.nan is invalid and will be ingorned during computation of np.mean()


def calculate_segmentation_metrics(true_labels, predicted_labels, number_classes=2, ignore_label=-1):
    if (true_labels == ignore_label).all():
        return [0] * 4

    true_labels = true_labels.flatten()
    predicted_labels = predicted_labels.flatten()
    valid_pix_ids = true_labels != ignore_label
    predicted_labels = predicted_labels[valid_pix_ids]
    true_labels = true_labels[valid_pix_ids]

    conf_mat = confusion_matrix(true_labels, predicted_labels, labels=list(range(number_classes)))
    norm_conf_mat = np.transpose(
        np.transpose(conf_mat) / conf_mat.astype(np.float).sum(axis=1))

    missing_class_mask = np.isnan(norm_conf_mat.sum(1))  # missing class will have NaN at corresponding class
    exsiting_class_mask = ~ missing_class_mask

    class_average_accuracy = nanmean(np.diagonal(norm_conf_mat))
    total_accuracy = (np.sum(np.diagonal(conf_mat)) / np.sum(conf_mat))
    ious = np.zeros(number_classes)
    for class_id in range(number_classes):
        ious[class_id] = (conf_mat[class_id, class_id] / (
                np.sum(conf_mat[class_id, :]) + np.sum(conf_mat[:, class_id]) -
                conf_mat[class_id, class_id]))
    miou = nanmean(ious)
    miou_valid_class = np.mean(ious[exsiting_class_mask])
    return miou, miou_valid_class, total_accuracy, class_average_accuracy, ious

def ddp_test_nerf(rank, args):
    ###### set up multi-processing
    setup(rank, args.world_size)
    ###### set up logger
    logger = logging.getLogger(__package__)
    setup_logger()

    ###### decide chunk size according to gpu memory
    if torch.cuda.get_device_properties(rank).total_memory / 1e9 > 14:
        logger.info('setting batch size according to 24G gpu')
        args.N_rand = 1024
        args.chunk_size = 8192
    else:
        logger.info('setting batch size according to 12G gpu')
        args.N_rand = 512
        args.chunk_size = 4096

    ###### create network and wrap in ddp; each process should do this
    start, models = create_nerf(rank, args)
    logits_2_label = lambda x: torch.argmax(torch.nn.functional.softmax(x, dim=-1), dim=-1)

    render_splits = [x.strip() for x in args.render_splits.strip().split(',')]
    # start testing
    for split in render_splits:
        out_dir = os.path.join(args.basedir, args.expname,
                               'render_{}_{:06d}'.format(split, start))
        if rank == 0:
            os.makedirs(out_dir, exist_ok=True)

        ###### load data and create ray samplers; each process should do this
        ray_samplers = load_data_split(args.datadir, args.scene, split, try_load_min_depth=args.load_min_depth)
        for idx in range(len(ray_samplers)):
            ### each process should do this; but only main process merges the results
            fname = '{:06d}.png'.format(idx)
            if ray_samplers[idx].img_path is not None:
                fname = os.path.basename(ray_samplers[idx].img_path)

            if os.path.isfile(os.path.join(out_dir, fname)):
                logger.info('Skipping {}'.format(fname))
                continue

            time0 = time.time()
            ret = render_single_image(rank, args.world_size, models, ray_samplers[idx], args.chunk_size)
            dt = time.time() - time0
            if rank == 0:    # only main process should do this
                logger.info('Rendered {} in {} seconds'.format(fname, dt))

                # only save last level
                im = ret[-1]['rgb'].numpy()
                # compute psnr if ground-truth is available
                if ray_samplers[idx].img_path is not None:
                    gt_im = ray_samplers[idx].get_img()
                    psnr = mse2psnr(np.mean((gt_im - im) * (gt_im - im)))
                    logger.info('{}: psnr={}'.format(fname, psnr))

                im = to8b(im)
                imageio.imwrite(os.path.join(out_dir, fname), im)

                im = logits_2_label(ret[-1]['semantic']).numpy()
                im = label_img_to_color(im)
                imageio.imwrite(os.path.join(out_dir, 'semantic_' + fname), im)

                im = logits_2_label(ret[-1]['bg_semantic']).numpy()
                im = label_img_to_color(im)
                imageio.imwrite(os.path.join(out_dir, 'bg_semantic_' + fname), im)

                im = logits_2_label(ret[-1]['fg_semantic']).numpy()
                im = label_img_to_color(im)
                imageio.imwrite(os.path.join(out_dir, 'fg_semantic_' + fname), im)

                im = ret[-1]['fg_rgb'].numpy()
                im = to8b(im)
                imageio.imwrite(os.path.join(out_dir, 'fg_' + fname), im)

                im = ret[-1]['bg_rgb'].numpy()
                im = to8b(im)
                imageio.imwrite(os.path.join(out_dir, 'bg_' + fname), im)

                im = ret[-1]['fg_depth'].numpy()
                im = colorize_np(im, cmap_name='jet', append_cbar=True)
                im = to8b(im)
                imageio.imwrite(os.path.join(out_dir, 'fg_depth_' + fname), im)

                im = ret[-1]['bg_depth'].numpy()
                im = colorize_np(im, cmap_name='jet', append_cbar=True)
                im = to8b(im)
                imageio.imwrite(os.path.join(out_dir, 'bg_depth_' + fname), im)

            torch.cuda.empty_cache()

    # clean up for multi-processing
    cleanup()


def test():
    parser = config_parser()
    args = parser.parse_args()
    logger.info(parser.format_values())

    if args.world_size == -1:
        args.world_size = torch.cuda.device_count()
        logger.info('Using # gpus: {}'.format(args.world_size))
    torch.multiprocessing.spawn(ddp_test_nerf,
                                args=(args,),
                                nprocs=args.world_size,
                                join=True)


if __name__ == '__main__':
    setup_logger()
    test()

