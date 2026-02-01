"""
Sample inference script with a pretrained UPLiFT model using uplift_extractor.py.

Example command:
python sample_inference.py --pretrained uplift_dinov3-splus16 --image imgs/Gigi_1_512.png --iters 4

Code by: Matthew Walmer and Anirud Aggarwal
"""
import os
import argparse
import random
import numpy as np
import torch
import torchvision.transforms.functional as F
from torchvision import transforms
from PIL import Image

from uplift import UPLiFTExtractor
from uplift.uplift_utils.pca_vis import pca

# Map pretrained names to hub_loader variants
PRETRAINED_TO_VARIANT = {
    'uplift_dinov2-s14': 'dinov2-s14',
    'uplift_dinov3-splus16': 'dinov3-splus16',
    'uplift_sd1.5vae': 'sd15-vae',
}


def main(args):
    assert args.image is not None
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    #####

    image = Image.open(args.image)

    if args.pretrained is not None:
        # Use hub_loader to download from HuggingFace
        from uplift.hub_loader import load_model
        variant = PRETRAINED_TO_VARIANT.get(args.pretrained, args.pretrained)
        UX = load_model(variant, pretrained=True, include_extractor=True, iters=args.iters,
                        out_size=args.outsize, return_base_feat=True, low_mem=args.low_mem)
    else:
        # Manual config/ckpt paths
        UX = UPLiFTExtractor(cfg_path=args.config, ckpt_path=args.ckpt, iters=args.iters,
                             out_size=args.outsize, return_base_feat=True, low_mem=args.low_mem)
    feat_uplift, feat = UX(image)

    #####

    image_t = F.pil_to_tensor(image)
    print('===')
    print('input image size:')
    print(image_t.shape)
    print('original backbone feature:')
    print(feat.shape)
    print('UPLiFT upsampled feature:')
    print(feat_uplift.shape)
    print('===')

    #####

    uname = 'uplift'
    if args.pretrained is not None:
        uname = args.pretrained

    # Diff Backbone: decode upsampled latents
    if UX.extractor_type == 'diff':
        print('Decoding upsampled latents to upsampled image')
        image_up = UX.decode(feat_uplift)
        image_out = os.path.basename(args.image) + '_%s-%i.png'%(uname, args.iters)
        print('Saving to: ' + image_out)
        image_up.save(image_out)

    # ViT Backbone: visualize latents with PCA
    else:
        print('Visualizing upsampled latents with PCA (may take >1 minute)')
        reduced_feats, _ = pca([feat_uplift, feat], dim=3)
        # uplift upsampled feature
        image_out = os.path.basename(args.image) + '_%s-%i-PCA.png'%(uname, args.iters)
        pca_feat_uplift = reduced_feats[0].squeeze(0)
        img = F.to_pil_image(pca_feat_uplift)
        img.save(image_out)
        print('saved: ' + image_out)
        # low res feature
        image_out = os.path.basename(args.image) + '_%s-base-feature-PCA.png'%(uname)
        pca_feat = reduced_feats[1].squeeze(0)
        pca_feat = F.resize(pca_feat, [pca_feat_uplift.shape[1], pca_feat_uplift.shape[2]], interpolation=transforms.InterpolationMode.NEAREST)
        img = F.to_pil_image(pca_feat)
        img.save(image_out)
        print('saved: ' + image_out)


#########################


def parse_args():
    parser = argparse.ArgumentParser('UPLiFT Inference Script')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--pretrained', type=str, default=None, help='selected a pretrained UPLiFT model by name')
    parser.add_argument('--config', type=str, default=None, help='or specify a path to an UPLiFT config file')
    parser.add_argument('--ckpt', type=str, default=None, help='and specify a path to an UPLiFT checkpoint file')
    parser.add_argument('--iters', type=int, default=1, help='number of UPLiFT upsampling steps to perform')
    parser.add_argument('--outsize', type=int, default=None, help='force a particular output size through resizing at the end')
    parser.add_argument('--image', type=str, default=None, help='provide a path to an image to run inference on. Will run inference instead of training')
    parser.add_argument('--low_mem', action='store_true', help='Enable/disable low-memory mode in LocalAttender, which sacrifices speed for lower max memory')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
