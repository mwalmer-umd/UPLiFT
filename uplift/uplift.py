"""
UPLiFT: Universal Pixel-dense Lightweight Feature Transforms

UPLiFT models can be used to efficiently upsample image features from a pretrained backbone to pixel-dense 
features, or other scales in between. UPLiFT builds on the LiFT-style approach for iterative feature upsampling,
but uses an improved architecture for better feature consistence through a new LocalAttender module. The UPLiFT
arch is specified using a config file, which provides flexible control over the model size and layers. Example
config files are provided in ./configs/

Code by: Matthew Walmer, Saksham Suri, and Anirud Aggarwal
"""
from typing import Callable
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TVF
from torchvision import transforms


########## UTILS ##########


# Add any shape residual x1 to x2 when channels may not match
# by performing truncation or zero-padding
def residual_auto(x1, x2):
    s1 = x1.shape[1]
    s2 = x2.shape[1]
    if s1 == s2:
        return x1 + x2
    elif s1 > s2: # truncation
        s = x2.shape[1]
        xr = x1[:,:s,:,:]
        return xr + x2
    else: # padding
        s = x1.shape[1]
        pad = torch.zeros_like(x2)
        pad[:,:s,:,:] = x1
        return pad + x2


# resize image_batch by a scaling factor
def resize_sf(image_batch, sf):
    d1 = int(image_batch.shape[2]*sf)
    d2 = int(image_batch.shape[3]*sf)
    image_res = TVF.resize(image_batch, [d1, d2])
    return image_res


'''
This LayerNorm code is reproduced from: https://github.com/facebookresearch/ConvNeXt
Copyright (c) Meta Platforms, Inc. and affiliates.
Distributed and used under an MIT License, as can be found in licenses/LayerNorm_LICENSE.txt
Please see the repository above for additional information.
'''
class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


########## STANDARD BLOCKS ##########


# placeholder or "no block"
class IdentityBlock(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


# simple block with a single conv operator, and an optional 1x1 conv
class SingleConv(nn.Module):
    def __init__(self, in_channels, h1, norm_type='batch', residual=True, stride=1, one_by=True):
        super().__init__()
        self.residual = residual
        self.norm_type = norm_type
        assert self.norm_type in ['batch','layer']
        self.stride = stride
        assert self.stride in [1,2] # optional strided conv
        self.one_by = one_by
        # Layers
        if self.one_by: # include 1x1 conv at end
            if self.norm_type == 'batch':
                self.conv = nn.Sequential(
                    nn.Conv2d(in_channels, h1, kernel_size=3, padding=1, bias=False, stride=stride),
                    nn.BatchNorm2d(h1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(h1, h1, kernel_size=1, bias=True)
                )
            else:
                self.conv = nn.Sequential(
                    nn.Conv2d(in_channels, h1, kernel_size=3, padding=1, bias=False, stride=stride),
                    LayerNorm(h1, eps=1e-6, data_format="channels_first"),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(h1, h1, kernel_size=1, bias=True)
                )
        else:
            if self.norm_type == 'batch':
                self.conv = nn.Sequential(
                    nn.Conv2d(in_channels, h1, kernel_size=3, padding=1, bias=False, stride=stride),
                    nn.BatchNorm2d(h1),
                    nn.ReLU(inplace=True)
                )
            else:
                self.conv = nn.Sequential(
                    nn.Conv2d(in_channels, h1, kernel_size=3, padding=1, bias=False, stride=stride),
                    LayerNorm(h1, eps=1e-6, data_format="channels_first"),
                    nn.ReLU(inplace=True)
                )
        # handle residual connection with strided conv
        self.res_down = None
        if self.residual and self.stride == 2:
            self.res_down = torch.nn.UpsamplingBilinear2d(scale_factor=0.5)

    def forward(self, x):
        x1 = self.conv(x)
        if self.residual:
            if self.stride == 2:
                xr = self.res_down(x)
            else:
                xr = x
            return residual_auto(xr, x1)
        else:
            return x1


# simple block with a two conv operators, and an optional 1x1 conv
class DoubleConv(nn.Module):
    def __init__(self, in_channels, h1, h2, norm_type='batch', residual=True, stride=1, one_by=True):
        super().__init__()
        self.residual = residual
        # norm
        self.norm_type = norm_type
        assert self.norm_type in ['batch','layer']
        self.stride = stride
        assert self.stride in [1,2] # optional strided conv
        self.one_by = one_by
        # Layers
        if self.one_by: # include 1x1 conv at end
            if self.norm_type == 'batch':
                self.conv = nn.Sequential(
                    nn.Conv2d(in_channels, h1, kernel_size=3, padding=1, bias=False, stride=stride),
                    nn.BatchNorm2d(h1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(h1, h2, kernel_size=3, padding=1, bias=False, stride=stride),
                    nn.BatchNorm2d(h2),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(h2, h2, kernel_size=1, bias=True)
                )
            else:
                self.conv = nn.Sequential(
                    nn.Conv2d(in_channels, h1, kernel_size=3, padding=1, bias=False, stride=stride),
                    LayerNorm(h1, eps=1e-6, data_format="channels_first"),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(h1, h2, kernel_size=3, padding=1, bias=False, stride=stride),
                    LayerNorm(h2, eps=1e-6, data_format="channels_first"),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(h2, h2, kernel_size=1, bias=True)
                )
        else:
            if self.norm_type == 'batch':
                self.conv = nn.Sequential(
                    nn.Conv2d(in_channels, h1, kernel_size=3, padding=1, bias=False, stride=stride),
                    nn.BatchNorm2d(h1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(h1, h2, kernel_size=3, padding=1, bias=False, stride=stride),
                    nn.BatchNorm2d(h2),
                    nn.ReLU(inplace=True)
                )
            else:
                self.conv = nn.Sequential(
                    nn.Conv2d(in_channels, h1, kernel_size=3, padding=1, bias=False, stride=stride),
                    LayerNorm(h1, eps=1e-6, data_format="channels_first"),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(h1, h2, kernel_size=3, padding=1, bias=False, stride=stride),
                    LayerNorm(h2, eps=1e-6, data_format="channels_first"),
                    nn.ReLU(inplace=True)
                )
        # handle residual connection with strided conv
        self.res_down = None
        if self.residual and self.stride == 2:
            self.res_down = torch.nn.UpsamplingBilinear2d(scale_factor=0.5)

    def forward(self, x):
        x1 = self.conv(x)
        if self.residual:
            if self.stride == 2:
                xr = self.res_down(x)
            else:
                xr = x
            return residual_auto(xr, x1)
        else:
            return x1


# Legacy block for replicating LiFT arch
class TransposeConv(nn.Module):
    def __init__(self, in_channels, h1, residual='none'):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, h1, kernel_size=2, stride=2)
        self.residual = residual
        assert self.residual in ['none','bilinear']
        if self.residual == 'bilinear':
            self.res_up = torch.nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x):
        x_up = self.up(x)
        if self.residual == 'bilinear':
            xr = self.res_up(x)
            x_up = residual_auto(xr, x_up)
        return x_up


class BlockGroup(nn.Module):
    def __init__(self, in_channels, cfg):
        super().__init__()
        self.cfg = cfg
        self.count = self.cfg.count
        assert cfg.block in ['IdentityBlock', 'SingleConv', 'DoubleConv', 'TransposeConv']
        if cfg.block in ['SingleConv', 'DoubleConv']:
            self.residual = getattr(self.cfg, 'residual', True)
            self.stride = getattr(self.cfg, 'stride', 1)
            self.one_by = getattr(self.cfg, 'one_by', True)
            self.norm = getattr(self.cfg, 'norm', 'batch')
            assert self.norm in ['batch','layer']
        # create blocks
        if self.count > 0:
            ic = in_channels
            blocks = []
            for i in range(cfg.count):
                oc = cfg.hc[i]
                if cfg.block == 'IdentityBlock':
                    cblk = IdentityBlock()
                elif cfg.block == 'SingleConv':
                    cblk = SingleConv(ic, oc, norm_type=self.norm, residual=self.residual, stride=self.stride, one_by=self.one_by)
                elif cfg.block == 'DoubleConv':
                    cblk = DoubleConv(ic, oc, oc, norm_type=self.norm, residual=self.residual, stride=self.stride, one_by=self.one_by)
                elif cfg.block == 'TransposeConv':
                    cblk = TransposeConv(ic, oc)
                else:
                    print('ERROR: invalid block type: ' + cfg.block)
                    exit(-1)
                blocks.append(cblk)
                ic = oc
            self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        if self.count < 1: return x
        return self.blocks(x)


########## UP/DOWN BLOCKS ##########


# downsampling block
class DownBlock(nn.Module):
    def __init__(self, in_channels, cfg):
        super().__init__()
        self.cfg = cfg
        self.down_op = cfg.down_op
        assert self.down_op in ['conv','max','none']
        if self.down_op == 'none':
            # option 1: no downsampling
            self.down = None
        else:
            self.factor = cfg.factor           
            # option 2: strided conv downsampling operator
            if self.down_op == 'conv':
                self.stride = int(1/self.factor)
                self.norm_op = nn.BatchNorm2d
                self.down = nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=self.stride),
                    self.norm_op(in_channels),
                    nn.ReLU(inplace=True),
                )
            # option 3: max pool downsampling
            elif self.down_op == 'max':
                self.down = nn.MaxPool2d(2, 2)
            # residual connection operator
            self.residual = cfg.residual
            assert self.residual in ['bilinear','none']
            if self.residual == 'bilinear':
                self.res_down = torch.nn.UpsamplingBilinear2d(scale_factor=self.factor)

    def forward(self, x):
        if self.down is None: return x
        x_down = self.down(x)
        if self.residual == 'bilinear':
            xr = self.res_down(x)
            x_down = residual_auto(xr, x_down)
        return x_down


# upsampling block
class UpBlock(nn.Module):
    def __init__(self, in_channels, cfg):
        super().__init__()
        self.cfg = cfg
        assert cfg.up_op in ['tconv','none']
        if cfg.up_op == 'none':
            # option 1: no upsampling
            self.up = None
        else:
            # option 2: transpose conv
            self.factor = cfg.factor            
            self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=self.factor, stride=self.factor)
            # residual connection operator
            self.residual = cfg.residual
            assert self.residual in ['bilinear','none']
            if cfg.residual == 'bilinear':
                self.res_up = torch.nn.UpsamplingBilinear2d(scale_factor=self.factor)

    def forward(self, x):
        if self.up is None: return x
        x_up = self.up(x)
        if self.residual == 'bilinear':
            xr = self.res_up(x)
            x_up = residual_auto(xr, x_up)
        return x_up


########## FEATURE COMBINER ##########


"""
concatenate multiple feature inputs: x = current features, f = encoder features / skip connections (optional), and 
noise channels (optional). Noise channels will be randomly generated if requested. ncat_c = number of noise channels
to add. ncat_s = scaling for noise magnitude
"""
def noise_and_combine(x, f=None, ncat_c=0, ncat_s=1, f_resample='none'):
    # (optional) append extra feature channels
    if f is not None:
        # (optional) feature resampling mode
        if f_resample == 'nearest':
            f = TVF.resize(f, [x.shape[2], x.shape[3]], interpolation=transforms.InterpolationMode.NEAREST)
        elif f_resample == 'bilinear':
            f = TVF.resize(f, [x.shape[2], x.shape[3]], interpolation=transforms.InterpolationMode.BILINEAR)
        x = torch.cat([x, f], dim=1)
    # (optional) generate and append noise channels
    if ncat_c > 0:
        ncat = ncat_s * torch.randn([x.shape[0], ncat_c, x.shape[2], x.shape[3]], device=x.device)
        x = torch.cat([x, ncat], dim=1)
    return x


########## UPLiFT IMAGE ENCODER MODULE ##########


class UPLiFTEncoder(nn.Module):
    def __init__(self, in_channels, cfg):
        super().__init__()
        self.cfg = cfg
        self.out_sizes = None
        self.down_steps = cfg.down_steps
        assert self.down_steps in [0,1,2,3,4] # supports 0 to 4 downsampling steps
        # modules
        ic = in_channels
        self.out_sizes = []
        # group 0
        self.block_0 = BlockGroup(ic, cfg.block_0)
        ic = cfg.block_0.hc[-1]
        self.out_sizes.append(ic)
        # group 1
        if self.down_steps >= 1:
            self.down_1 = DownBlock(ic, cfg.down_1)
            self.block_1 = BlockGroup(ic, cfg.block_1)
            ic = cfg.block_1.hc[-1]
            self.out_sizes.append(ic)
        # group 2
        if self.down_steps >= 2:
            self.down_2 = DownBlock(ic, cfg.down_2)
            self.block_2 = BlockGroup(ic, cfg.block_2)
            ic = cfg.block_2.hc[-1]
            self.out_sizes.append(ic)
        # group 3
        if self.down_steps >= 3:
            self.down_3 = DownBlock(ic, cfg.down_3)
            self.block_3 = BlockGroup(ic, cfg.block_3)
            ic = cfg.block_3.hc[-1]
            self.out_sizes.append(ic)
        # group 4
        if self.down_steps >= 4:
            self.down_4 = DownBlock(ic, cfg.down_4)
            self.block_4 = BlockGroup(ic, cfg.block_4)
            ic = cfg.block_4.hc[-1]
            self.out_sizes.append(ic)


    def get_out_sizes(self):
        return self.out_sizes


    def forward(self, x):
        ret = []
        # group 0
        x = self.block_0(x)
        ret.append(x)
        # group 1
        if self.down_steps >= 1:
            x = self.down_1(x)
            x = self.block_1(x)
            ret.append(x)
        # group 2
        if self.down_steps >= 2:
            x = self.down_2(x)
            x = self.block_2(x)
            ret.append(x)
        # group 3
        if self.down_steps >= 3:
            x = self.down_3(x)
            x = self.block_3(x)
            ret.append(x)
        # group 4
        if self.down_steps >= 4:
            x = self.down_4(x)
            x = self.block_4(x)
            ret.append(x)
        return ret


########## UPLiFT DECODER/UPSAMPLER MODULE ##########


class UPLiFTDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, cfg, enc_channels=None, using_attender=False):
        super().__init__()
        self.cfg = cfg
        self.ncat_c = cfg.ncat_c
        self.ncat_s = cfg.ncat_s
        self.up_steps = cfg.up_steps
        self.using_attender = using_attender
        assert self.up_steps in [0,1,2,3,4] # supports 0 to 4 upsampling steps
        # encoder connection inputs
        if enc_channels is None:
            self.enc_channels = [0,0,0,0,0] # no encoder inputs
        else:
            self.enc_channels = enc_channels
        # encoder connection resampling mode: downsample the encoder latent connections to the needed size
        self.enc_resample = getattr(self.cfg, 'enc_resample', 'none')
        assert self.enc_resample in ['none','nearest','bilinear']
        # modules
        # Group 0
        ic = in_channels + self.enc_channels[-1] + cfg.ncat_c
        self.block_0 = BlockGroup(ic, cfg.block_0)
        ic = cfg.block_0.hc[-1]
        # Group 1
        if self.up_steps >= 1:
            self.up_1 = UpBlock(ic, cfg.up_1)
            ic = ic + self.enc_channels[-2] + cfg.ncat_c
            self.block_1 = BlockGroup(ic, cfg.block_1)
            ic = cfg.block_1.hc[-1]
        # Group 2
        if self.up_steps >= 2:
            self.up_2 = UpBlock(ic, cfg.up_2)
            ic = ic + self.enc_channels[-3] + cfg.ncat_c
            self.block_2 = BlockGroup(ic, cfg.block_2)
            ic = cfg.block_2.hc[-1]
        # Group 3
        if self.up_steps >= 3:
            self.up_3 = UpBlock(ic, cfg.up_3)
            ic = ic + self.enc_channels[-4] + cfg.ncat_c
            self.block_3 = BlockGroup(ic, cfg.block_3)
            ic = cfg.block_3.hc[-1]
        # Group 4
        if self.up_steps >= 4:
            self.up_4 = UpBlock(ic, cfg.up_4)
            ic = ic + self.enc_channels[-5] + cfg.ncat_c
            self.block_4 = BlockGroup(ic, cfg.block_4)
            ic = cfg.block_4.hc[-1]
        self.out_channels = ic
        # Output reshaping layer
        if not self.using_attender:
            self.outc = nn.Conv2d(ic, out_channels, kernel_size=1)
            self.residual_outc = getattr(self.cfg, 'residual_outc', True)
            self.out_channels = out_channels


    def forward(self, x, enc_feats):
        # Group 0
        x = noise_and_combine(x, enc_feats[-1], self.ncat_c, self.ncat_s, self.enc_resample)
        x = self.block_0(x)
        # Group 1
        if self.up_steps > 0:
            x = self.up_1(x)
            x = noise_and_combine(x, enc_feats[-2], self.ncat_c, self.ncat_s, self.enc_resample)
            x = self.block_1(x)
        # Group 2
        if self.up_steps > 1:
            x = self.up_2(x)
            x = noise_and_combine(x, enc_feats[-3], self.ncat_c, self.ncat_s, self.enc_resample)
            x = self.block_2(x)
        # Group 3
        if self.up_steps > 2:
            x = self.up_3(x)
            x = noise_and_combine(x, enc_feats[-4], self.ncat_c, self.ncat_s, self.enc_resample)
            x = self.block_3(x)
        # Group 4
        if self.up_steps > 3:
            x = self.up_4(x)
            x = noise_and_combine(x, enc_feats[-5], self.ncat_c, self.ncat_s, self.enc_resample)
            x = self.block_4(x)
        # Final
        if self.using_attender: return x
        logits = self.outc(x)
        if self.residual_outc:
            logits = residual_auto(x, logits) # residual connection
        return logits


########## UPLiFT ##########


class UPLiFT(nn.Module):
    def __init__(self, in_channels, patch_size, cfg, dtype: torch.dtype = torch.float32, low_mem=False):
        super().__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.cfg = cfg
        self.up_factor = self.cfg.up_factor
        self.target_dtype = dtype
        self.low_mem = low_mem

        # attender formulation
        self.using_attender = False
        if hasattr(self.cfg, 'attender') and self.cfg.attender != -1:
            self.using_attender = True
        if not self.using_attender and self.low_mem:
            print('WARNING: UPLiFT low-mem mode only applies when LocalAttender is used')

        # modules
        self.encoder = UPLiFTEncoder(3, self.cfg.encoder) # note: assumes image channels == 3
        enc_channels = self.encoder.get_out_sizes()
        self.decoder = UPLiFTDecoder(self.in_channels, self.in_channels, self.cfg.decoder, enc_channels, using_attender=self.using_attender)
        self.attender = None
        if self.using_attender:
            self.attender = LocalAttender(self.decoder.out_channels, self.cfg.attender, low_mem=self.low_mem)

        # encoder sharing mode: when enabled and when running iterative inference, will use a single encoder pass for multiple decoder steps
        self.enc_share = getattr(self.cfg, 'enc_share', False)
        
        # (optional) refiner block, applied after the attender block
        self.refiner = None
        if hasattr(self.cfg, 'refiner'):
            self.ref_ncat_c = self.cfg.refiner.ncat_c
            self.ref_ncat_s = self.cfg.refiner.ncat_s
            self.ref_enc_resample = self.cfg.refiner.enc_resample
            if hasattr(self.cfg.refiner, 'ref_con'):
                self.ref_con = self.cfg.refiner.ref_con
            else:
                self.ref_con = -1 - self.cfg.decoder.up_steps
            ref_ic = in_channels + self.cfg.refiner.ncat_c + enc_channels[self.ref_con]
            self.refiner = BlockGroup(ref_ic, self.cfg.refiner.block_0)
            self.ref_outc = nn.Conv2d(self.cfg.refiner.block_0.hc[-1], self.in_channels, kernel_size=1)

        # iterative inference support
        self.iters = 1
        self.return_all_steps = False

        # move to desired dtype
        self._apply_dtype()
        

    # enable iterative mode
    def enable_iterative(self, iters):
        assert iters > 0
        self.iters = iters


    # enable returning intermediate latents for iterative runs
    def return_steps(self, setting):
        self.return_all_steps = setting


    # single upsampling step
    def upsample(self, img, x, cur_iter=0, ex_feats=None):
        x_in = x
        # UPLiFT Encoder
        if not self.enc_share or ex_feats is None: # compute encoder features
            # pre-scaler to support ViTs with patch size 14
            if self.patch_size == 14:
                d2 = (img.shape[2]*16) // 14
                d3 = (img.shape[3]*16) // 14
                img = TVF.resize(img, [d2, d3])
            enc_feats = self.encoder(img)
        else: # encoder feature re-use
            enc_feats = ex_feats
        # UPLiFT Decoder
        x = self.decoder(x, enc_feats)
        # Local Attender
        if self.attender is not None:
            x = self.attender(x, x_in)
        # (optional) post-attender Refiner Block
        if self.refiner is not None:
            x = noise_and_combine(x, enc_feats[self.ref_con], self.ref_ncat_c, self.ref_ncat_s, self.ref_enc_resample)
            x = self.refiner(x)
            x = self.ref_outc(x)
        return x, enc_feats


    def forward(self, img, x, outsize=None):
        # track itermediate steps
        x_steps = []
        # single or iterative upsampling
        ex_feats = None
        for i in range(self.iters):
            if i>0 and not self.enc_share:
                img_res = resize_sf(img, self.up_factor**i)
            else:
                img_res = img
            x, ex_feats = self.upsample(img_res, x, i, ex_feats)
            x_steps.append(x)
        # (optional) rescale final output to a desired size if specified, overrides self.return_all_steps
        if outsize is not None:
            x = TVF.resize(x, outsize, interpolation=transforms.InterpolationMode.BILINEAR)
            return x
        # (optional) Return intermediate steps
        if self.return_all_steps:
            return x_steps
        return x


    def _apply_dtype(self):
        """Move model parameters and buffers to self.target_dtype."""
        for module in self.modules():
            for param in module.parameters(recurse=False):
                param.data = param.data.to(self.target_dtype)
            for buf_name, buf in module._buffers.items():
                if buf is not None:
                    module._buffers[buf_name] = buf.to(self.target_dtype)


    def compiled(
        self,
        backend: str,
        mode: str,
        fullgraph: bool,
        dynamic: bool,
        device: str | torch.device = "cuda",
        set_tf32: bool = True,
        warmup: Callable | None = None,
    ) -> "nn.Module":
        """
        Fully compile UPLiFT for inference.

        Args:
            device: target device to place the module before compile.
            backend: torch.compile backend (default: 'inductor').
            mode: compile mode (e.g., 'default', 'reduce-overhead', 'max-autotune').
            fullgraph: require a single graph (True gives best fusion; set False if you hit graph breaks).
            dynamic: allow some dynamic shape handling (slightly less aggressive, but robust).
            set_tf32: enable TF32 on CUDA (good perf on Ampere+).
            warmup: optional callable with signature `() -> tuple[tuple,args], dict[kwargs]`
                    that returns inputs (args, kwargs) to run a single dry-forward on the compiled module.

        Returns:
            The compiled module.
        """
        # 1) Device + dtype + eval
        if device is not None:
            super().to(device)
        # re-assert param/buffer dtype in case weights were loaded on CPU, etc.
        self._apply_dtype()
        self.eval()

        # 2) Perf knobs, free performance when supported
        if set_tf32 and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # 3) Compile the whole module
        compiled: nn.Module = torch.compile(  # type: ignore
            self,
            backend=backend,
            mode=mode,
            fullgraph=fullgraph,
            dynamic=dynamic,
        )

        # 4) Optional single warmup call to trigger compilation ahead of time
        if warmup is not None:
            with torch.inference_mode():
                args, kwargs = warmup()
                _ = compiled(*args, **kwargs)

        return compiled


########## LOCAL ATTENDER ##########


"""
Local Attender approach for efficient local-attention-based feature pooling,
without requiring Queries and Keys or quadratic running time. Uses attention
pooling over a fixed local neighborhood, defined by a set of positional offsets.
Local Attender uses a high-resolution "guide" feature  map to guide the upsampling
of a low-resolution "value" feature map. Through local attentional pooling, the
output feature map will maintain a similar feature distribution to the input "value"
feature map.

SETTINGS:
in_channels   - The channel depth of the "guide" feature map input
num_connected - The size of the local neighborhood to attend over
conv_res      - Enable/disable residual connection on 1x1 conv layer
low_mem       - Enable/disable low-memory mode, which sacrifices speed for lower max memory

Guide high-res feature map input is of shape:
[Batch, H_high, W_high, C_g]

Value low-res feature map input is of shape:
[Batch, H_low, W_low, C_v]
H_out and W_out must be integer multiples of H and W

Output map is of shape:
[Batch, H_high, W_high, C_v]
"""
class LocalAttender(nn.Module):
    def __init__(self, in_channels, num_connected=5, conv_res=True, low_mem=False):
        super().__init__()
        self.in_channels = in_channels
        self.num_connected = num_connected
        # select offset set based on num_connected
        if self.num_connected == 5:
            self.offsets = [(-1,0),(0,-1),(0,0),(0,1),(1,0)]
            self.pn = 1 # padding number
        elif self.num_connected == 9:
            self.offsets = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,0),(0,1),(1,-1),(1,0),(1,1)]
            self.pn = 1
        elif self.num_connected == 13:
            self.offsets = [(-2,0),(-1,-1),(-1,0),(-1,1),(0,-2),(0,-1),(0,0),(0,1),(0,2),(1,-1),(1,0),(1,1),(2,0)]
            self.pn = 2
        elif self.num_connected == 17:
            self.offsets = [(-2,-2),(-2,0),(-2,2),(-1,-1),(-1,0),(-1,1),(0,-2),(0,-1),(0,0),(0,1),(0,2),(1,-1),(1,0),(1,1),(2,-2),(2,0),(2,2)]
            self.pn = 2
        elif self.num_connected == 25:
            self.offsets = [(-2,-2),(-2,-1),(-2,0),(-2,1),(-2,2),(-1,-2),(-1,-1),(-1,0),(-1,1),(-1,2),(0,-2),(0,-1),(0,0),(0,1),(0,2),(1,-2),(1,-1),(1,0),(1,1),(1,2),(2,-2),(2,-1),(2,0),(2,1),(2,2)]
            self.pn = 2
        else:
            print('ERROR: LocalAttender invalid num_connected')
            exit(-1)
        # attender map maker
        self.conv1 = nn.Conv2d(in_channels, num_connected, kernel_size=1)
        self.conv_res = conv_res
        # padding
        self.pad = nn.ReplicationPad2d(self.pn)
        # low mem mode
        self.low_mem = low_mem


    # visualize the offset map for the given setting
    def show_offsets(self):
        k = 3 + (2*self.pn)
        b = self.pn + 1
        vis = np.zeros([k,k])
        for off in self.offsets:
            h, w = off
            vis[b+h, b+w] = 1
        print(vis)


    def make_offsets(self, x):
        H = x.shape[2]
        W = x.shape[3]
        # [B, C, H, W]
        x = self.pad(x)
        # [B, C, H+(2*PN), W+(2*PN)]
        x_all = []
        for off in self.offsets:
            off_0, off_1 = off
            x_off = x[:, :, self.pn+off_0:self.pn+off_0+H, self.pn+off_1:self.pn+off_1+W]
            # [B, C, H, W]
            x_all.append(x_off)
        x = torch.stack(x_all, dim=2)
        # [B, C, D, H, W]
        return x


    def forward(self, guide, value):
        # value shape: [B, C, H, W]
        x = value
        B = x.shape[0]
        C = x.shape[1]
        H = x.shape[2]
        W = x.shape[3]

        # create attender map with shape: [B, D, H_out, W_out]
        att = self.conv1(guide)
        if self.conv_res:
            att = residual_auto(guide, att)
        D = att.shape[1]
        H_out = att.shape[2]
        W_out = att.shape[3]

        # identify integer scaling factor
        I = att.shape[2] // x.shape[2]
        assert I * x.shape[2] == att.shape[2]
        assert I * x.shape[3] == att.shape[3]

        # handle value map
        x = self.make_offsets(x) # [B, C, H, W] -> [B, C, D, H, W]
        x = torch.unsqueeze(x, dim=4) # -> [B, C, D, H, 1, W]
        x = torch.unsqueeze(x, dim=6) # -> [B, C, D, H, 1, W, 1]
        
        # handle attender map
        att = F.softmax(att, dim=1) # [B, D, H_out, W_out]
        att = torch.reshape(att, [B, D, H, I, W, I]) # -> [B, D, H, I, W, I]
        att= torch.unsqueeze(att, dim=1) # -> [B, 1, D, H, I, W, I]

        # pool features
        if not self.low_mem:
            # normal mode - parallel pooling
            res = x * att # -> [B, C, D, H, I, W, I]
            res = torch.sum(res, dim=2) # -> [B, C, H, I, W, I]
            res = res.reshape([B, C, H_out, W_out]) # -> [B, C, H_out, W_out]
        else:
            # low-mem mode - sequential pooling
            res = torch.zeros([B,C,H_out,W_out]).to(x.device)
            for d in range(self.num_connected):
                x_d = x[:,:,d,:,:,:,:]
                att_d = att[:,:,d,:,:,:,:]
                res_d = x_d * att_d
                res_d = res_d.reshape([B, C, H_out, W_out])
                res += res_d
        return res