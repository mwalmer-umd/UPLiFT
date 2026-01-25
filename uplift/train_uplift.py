"""
Train an UPLiFT module using the Backbone and Dataset specified in the given config file. Supports different 
types of backbones including representation-learning-focused backbones and auto-encoder backbones for diffusion
models.

Example command:
python train_uplift.py --config configs/uplift_dinov2-s14.yaml

Code by: Matthew Walmer and Saksham Suri
"""
import os
import argparse
import random
import time
import shutil
from glob import glob
from collections import OrderedDict

from omegaconf import OmegaConf
import natsort
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as TVF
from torchvision import datasets, transforms
from PIL import Image

from uplift.extractors.diff_extractor import DiffExtractor, bytes_to_giga_bytes
from uplift.extractors.vit_wrapper import PretrainedViTWrapper
from uplift.datasets.datasets_helper import get_dataroot
from uplift.uplift_utils.general_utils import *
from uplift.uplift_utils.data_utils import ImageFolderWithPaths, SingleFolderWithPaths
from uplift.uplift_utils.cache_manager import CacheManager
from uplift.uplift import UPLiFT


##############################


def main(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ##### LOGDIR #####

    # prepare logdir
    if args.name is None:
        # if experiment name not specified, default to basename of config
        args.name = os.path.basename(args.config).replace('.yaml','')
    output_dir = os.path.join(args.logdir, args.name)
    os.makedirs(output_dir, exist_ok=True)
    print("Output dir: ", output_dir)

    ##### CONFIG #####

    # check for existing config
    write_conf = True
    files = os.listdir(output_dir)
    for f in files:
        if '.yaml' in f:
            ex_conf = os.path.join(output_dir, f)
            print('found existing config file at: ' + ex_conf)
            if args.config is not None:
                # option 1: replace existing config
                if args.force:
                    print('WARNING: --force mode enabled, existing config will be replaced')
                    os.remove(ex_conf)
                # option 2: using existing config, overriding --config setting
                else:
                    print('overriding --config setting: ' + args.config)
                    args.config = ex_conf
                    write_conf = False
            else:
                # existing config will be used
                args.config = ex_conf
                write_conf = False
            break
    if write_conf:
        # copy config file to experiment dir
        bn = os.path.basename(args.config)
        out_conf = os.path.join(output_dir, bn)
        shutil.copyfile(args.config, out_conf)
    # load config file
    cfg = OmegaConf.load(args.config)

    # check settings
    assert cfg.train.down_steps > 0

    ##### BACKBONE #####

    # identify backbone name, prepare backbone feature extractor
    diff_backbone = (cfg.backbone.diff_pipe_name is not None)
    if diff_backbone: # hugging-face diffusion pipeline backbones
        backbone_name = cfg.backbone.diff_pipe_name.split('/')[-1] + '_vae'
        extractor = DiffExtractor(cfg.backbone.diff_pipe_name, device)
    else: # ViT feature extractor
        backbone_name = cfg.backbone.model_type
        extractor = PretrainedViTWrapper(cfg.backbone.model_type).to(device)
    
    ##### UPLiFT #####

    # prepare UPLiFT
    uplift = UPLiFT(cfg.backbone.channel, cfg.backbone.patch, cfg.uplift)
    uplift.return_steps(True) # for intermediate loss terms
    pc = count_parameters(uplift)
    if args.show: # --show mode: print UPLiFT arch and exit
        print(uplift)
        print('params: ' + str(pc))
        exit()
    else:
        print('params: ' + str(pc))
    # Data Parallel Multi-GPU training - recommended only for use with an existing feature cache
    using_dpar = False
    if args.dpar and torch.cuda.device_count() > 1:
        using_dpar = True
        print('Using %i GPUs'%torch.cuda.device_count())
        uplift = nn.DataParallel(uplift)
    uplift = uplift.to(device)

    ##### DATASET #####

    # prepare dataset
    dataroot = get_dataroot(cfg.data.dataset)
    print('LOADING DATASET: %s (%s)'%(cfg.data.dataset, dataroot))
    # loaders for diffusion backbones running with DiffExtractor
    if diff_backbone:   
        if 'imagenet' in cfg.data.dataset:
            train_dataset = ImageFolderWithPaths(dataroot, transform=transforms.Compose([
                transforms.v2.RGB(),
                transforms.Resize(cfg.train.imsize, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(cfg.train.imsize),
                transforms.ToTensor(),
            ]))
        else:
            train_dataset = SingleFolderWithPaths(dataroot, transform=transforms.Compose([
                transforms.v2.RGB(),
                transforms.Resize(cfg.train.imsize, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(cfg.train.imsize),
                transforms.ToTensor(),
            ]))
    # loaders for ViT backbones like DINOv2
    else:
        mean = (0.485, 0.456, 0.406) if "dino" in cfg.backbone.model_type else (0.5, 0.5, 0.5)
        std = (0.229, 0.224, 0.225) if "dino" in cfg.backbone.model_type else (0.5, 0.5, 0.5) 
        if 'imagenet' in dataroot:
            train_dataset = ImageFolderWithPaths(dataroot, transform=transforms.Compose([
                transforms.Resize((cfg.train.imsize, cfg.train.imsize)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]))
        else:
            train_dataset = SingleFolderWithPaths(dataroot, transform=transforms.Compose([
                transforms.v2.RGB(),
                transforms.Resize(cfg.train.imsize, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(cfg.train.imsize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]))
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=True, num_workers=4)

    # enable/disable UPLiFT encoder preprocessing
    no_UEP = False
    if hasattr(cfg.uplift, 'no_UEP'):
        no_UEP = cfg.uplift.no_UEP
        if no_UEP:
            if not diff_backbone:
                print('ERROR: no_UEP mode can only be used with a diffusion backbone')
                exit(-1)
            else:
                print('no_UEP mode enabled. Raw images will be fed to the UPLiFT encoder')

    # (optional) cache manager to accelerate training
    cache_manager = None
    if cfg.data.cache_dir is not None and not args.nocache:
        cache_manager = CacheManager(cfg.data.cache_dir, cfg.data.dataset, dataroot, backbone_name, device)

    ##### OPTIMIZER #####

    # define optimizer
    optimizer = torch.optim.Adam(uplift.parameters(), lr=cfg.train.lr)

    # define reconstruction loss
    assert cfg.train.loss in ['cosine','l1','l2']
    if cfg.train.loss == 'l1':
        criterion = torch.nn.L1Loss()
    elif cfg.train.loss == 'l2':
        criterion = torch.nn.MSELoss()
    else:
        criterion = torch.nn.CosineEmbeddingLoss()

    # (optional) use only deepest training depth
    only_deepest = getattr(cfg.train, 'only_deepest', False)
    if only_deepest:
        print('ONLY DEEPEST MODE ENABLED')

    ##### LOAD CHECKPOINT #####

    # if model exists in dir, load the latest epoch model
    epoch_start = 0
    if len(glob(os.path.join(output_dir, 'uplift_*.pth'))) > 0:
        model_paths = glob(os.path.join(output_dir, 'uplift_*.pth'))
        model_paths = natsort.natsorted(model_paths)
        print('loading checkpoint file: ' + model_paths[-1])
        loaded_state_dict = torch.load(model_paths[-1])
        # check for compatibility
        if not using_dpar:
            dpar_warning = False
            k_list = list(loaded_state_dict.keys())
            for k in k_list:
                if 'module.' in k:
                    dpar_warning = True
                    k_new = k.replace('module.','')
                    loaded_state_dict[k_new] = loaded_state_dict[k]
                    del loaded_state_dict[k]
            if dpar_warning:
                print('WARNING: This UPLiFT model checkpoint was trained with DataParallel. state_dict keys have been renamed to run without it.')
        # load
        uplift.load_state_dict(loaded_state_dict)
        epoch_start = int(model_paths[-1].split('_')[-1].split('.')[0])
        print("Loaded UPLiFT model from epoch {}".format(epoch_start))

    ##### TRAIN #####

    for epoch in range(epoch_start, cfg.train.epochs):
        epoch_start_time = time.time()
        for batch_index, image_data in enumerate(dataloader):
            batch_start_time = time.time()

            # data unpacking
            if 'imagenet' in cfg.data.dataset:
                image_batch = image_data[0].to(device)
                image_paths = image_data[2]
            else:
                image_batch = image_data[0].to(device)
                image_paths = image_data[1]

            # image downsampling for each running depth
            image_scales = OrderedDict()
            image_scales[cfg.train.imsize] = image_batch
            cursize = cfg.train.imsize
            for i in range(cfg.train.down_steps):
                # target scale
                if cursize not in image_scales:
                    image_res = TVF.resize(image_batch, [cursize, cursize])
                    image_scales[cursize] = image_res
                # input scale
                insize = int(cursize / cfg.uplift.up_factor)
                if insize not in image_scales:
                    image_res = TVF.resize(image_batch, [insize, insize])
                    image_scales[insize] = image_res
                # next step
                cursize = int(cursize/cfg.uplift.up_factor)

            # feature extraction and optional caching
            feat_scales = {}
            for s_i, s in enumerate(image_scales.keys()):
                # load cached feat
                if cache_manager is not None:
                    feat = cache_manager.load_cache(image_paths, s)
                    if feat is not None:
                        feat_scales[s] = feat
                        continue
                # generate feat
                image_in = image_scales[s]
                with torch.no_grad():
                    if diff_backbone:
                        feat = extractor(image_in)
                    else:
                        pred = extractor(image_in)
                        feat = pred[0]
                feat_scales[s] = feat
                # cache feat
                if cache_manager is not None:
                    cache_manager.save_cache(image_paths, feat, s)
                if args.feat: # show feature mode: display feature and stop
                    print('image_in')
                    print(image_in.shape)
                    print('feature')
                    print(feat.shape)
                    exit()

            # select running levels
            run_levels = np.arange(0, cfg.train.down_steps)
            if only_deepest: # (optional) only run deepest depth
                run_levels = [cfg.train.down_steps-1]

            # uplift feature upsampling and loss computation
            loss = 0
            for i in run_levels:
                # select input and target scales based on run level
                scale_tar = int(cfg.train.imsize/(cfg.uplift.up_factor**i))
                scale_in = int(scale_tar/cfg.uplift.up_factor)
                feat_in = feat_scales[scale_in]
                
                # (optional) feature noising augmentation
                if cfg.train.lat_aug > 0:
                    noise_aug = cfg.train.lat_aug * torch.randn(feat_in.shape, device=feat_in.device)
                    feat_in += noise_aug

                # image pre-processing
                image_orig = image_scales[scale_in]
                if diff_backbone and not no_UEP:
                    image_orig = extractor.preproc(image_orig)
                image_in = image_orig

                # Upsample with UPLiFT
                if cfg.train.multi_step:
                    iter_mode = i+1
                    uplift.enable_iterative(iter_mode)
                feats_out = uplift(image_in, feat_in)

                # Compute losses
                for f_i, feat_out in enumerate(feats_out):
                    # get gt feature
                    scale_dst = feat_out.shape[2] * cfg.backbone.patch
                    feat_tar = feat_scales[scale_dst]
                    # loss
                    if cfg.train.loss == 'cosine':
                        feat_out_re = reshape_for_cos(feat_out)
                        feat_tar_re = reshape_for_cos(feat_tar)
                        cur_loss = criterion(feat_out_re, feat_tar_re, torch.ones(feat_out_re.shape[0]).to(device))
                    else:
                        cur_loss = criterion(feat_out, feat_tar)
                    loss += cur_loss

            # optimizer step
            print('Epoch [{}/{}], Batch [{}/{}], Loss: {:.4f}'.format(epoch + 1, cfg.train.epochs, batch_index + 1, len(dataloader), loss.item()),flush=True)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print('Time taken for 1 batch: {} seconds'.format(time.time() - batch_start_time))
            if args.mem:
                print(f"Max memory allocated: {bytes_to_giga_bytes(torch.cuda.max_memory_allocated())} GB")
                print(f"Max memory reserved:  {bytes_to_giga_bytes(torch.cuda.max_memory_reserved())} GB")
            
        print('Time taken for 1 epoch: {} seconds'.format(time.time() - epoch_start_time))

        # save checkpoints and optional eval visualizations
        if (epoch + 1) % cfg.train.save_every == 0 and cfg.train.epochs > 1:
            torch.save(uplift.state_dict(), os.path.join(output_dir, 'uplift_{}.pth'.format(epoch + 1)))

            # (optional) limit number of recent checkpoints to keep
            if cfg.train.keep_n > 0:
                limit_checkpoints(output_dir, cfg.train.keep_n, 'uplift_')

            # (optional) visualize auto-encoding upsampling, for VAE backbone only
            if diff_backbone and cfg.data.val_imgs is not None:
                if not os.path.isdir(cfg.data.val_imgs):
                    print('WARNING: could not find: ' + cfg.data.val_imgs)
                    print('visualizations will be skipped')
                else:
                    uplift.eval()
                    if cfg.train.multi_step:
                        uplift.enable_iterative(1)
                    val_files = os.listdir(cfg.data.val_imgs)
                    with torch.no_grad():
                        for vf in val_files:
                            image_file = os.path.join(cfg.data.val_imgs, vf)
                            img = Image.open(image_file)
                            # extract backbone features
                            lat = extractor(img)
                            # apply UPLiFT
                            if not no_UEP:
                                img = extractor.preproc(img)
                            else: # no_UEP mode: do not run extra preproc if no_UEP mode is on
                                img = TVF.to_tensor(img).to(device)
                                img = torch.unsqueeze(img, 0)
                            lat = uplift(img, lat)[-1]
                            # decode image
                            img = extractor.decode(lat)
                            # save image
                            outname = os.path.join(output_dir, '%s_uplift_%03i.png'%(vf, epoch + 1))
                            img.save(outname)
                    uplift.train()
                    del lat, img
                    torch.cuda.empty_cache()
    
    # save final model
    torch.save(uplift.state_dict(), os.path.join(output_dir, 'uplift.pth'))


##############################


def parse_args():
    parser = argparse.ArgumentParser('Train an UPLiFT model')
    ### CONFIG ###
    parser.add_argument('--config', type=str, default=None, help='path to config file')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--logdir', type=str, default='experiments', help='base dir to save experiment logs to')
    parser.add_argument('--name', type=str, default=None, help='name of experiment / dir to save logs to. If not specified, will default to the config name')
    ### SPECIAL ###
    parser.add_argument('--force', action='store_true', help='Force override and overwrite the config in the destination dir')
    parser.add_argument('--mem', action='store_true', help='Report GPU memory used after each step')
    parser.add_argument('--show', action='store_true', help='Show the UPLiFT model arch and stop')
    parser.add_argument('--feat', action='store_true', help='extract the first feature, show it, and stop')
    parser.add_argument('--nocache', action='store_true', help='Force disable feature cache')
    parser.add_argument('--dpar', action='store_true', help='enable DataParallel if multiple GPUs are available')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
