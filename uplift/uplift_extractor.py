"""
Inference wrapper to help load a pretrained UPLiFT model and its associated visual
feature backbone to easily extract high-resolution image features. Example usage can
be seen in sample_inference.py.

Code by: Matthew Walmer and Anirud Aggarwal
"""
import os
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from omegaconf import OmegaConf

from uplift.uplift import UPLiFT


'''
Wrapper for Feature Extraction with UPLiFT. Will also load the appropriate feature extracting backbone based on the config. You
must specify a path to an UPLiFT config file and a pretrained checkpoint file to load. Alternately, --pretrained can be used to
simplify loading of the provided pretrained models.

SETTINGS:
pretrained       - Select one of several pretrained models by name
cfg_path         - When not using --pretrained, you must manually select a path to an UPLiFT config file
ckpt_path        - When not using --pretrained, you must manually select a path to an UPLiFT checkpoint file
iters            - How many upsampling steps to perform with the given UPLiFT. Set this number depending on the scaling factor
                       and patch size to produce pixel-dense features.
out_size         - Force a specific output size. If the base output size does not match, it will be bilinearly resampled to out_size
no_transform     - Set to True to disable all preprocessing, for cases where preproc is handled externally already
return_steps     - Return the intermediate upsampling steps too
return_base_feat - Return the base feature from the extractor backbone too
silent           - Disable print statements
auto_resize      - Automatically resize inputs if they are not a multiple of the backbone patch size
low_mem          - Enable/disable low-memory mode in LocalAttender, which sacrifices speed for lower max memory
'''
class UPLiFTExtractor(nn.Module):
    def __init__(self, pretrained=None, cfg_path=None, ckpt_path=None, config=None, weights=None,
            iters=1, out_size=None, no_transform=False, return_steps=False, return_base_feat=False,
            silent=False, auto_resize=True, low_mem=False):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.iters = iters
        self.out_size = out_size
        self.no_transform = no_transform
        self.return_steps = return_steps
        self.return_base_feat = return_base_feat
        self.silent = silent
        self.auto_resize = auto_resize
        self.low_mem = low_mem

        ##### CONFIG #####
        # Hub loader path: config and weights provided directly
        if config is not None:
            self.cfg_path = config
            self.ckpt_path = weights
            self.cfg = OmegaConf.load(self.cfg_path)
        # Legacy pretrained path: lookup by name
        elif pretrained is not None:
            assert pretrained in ['uplift_dinov2-s14', 'uplift_dinov3-splus16', 'uplift_sd1.5vae']
            pkg_root = Path(__file__).parent
            self.cfg_path = str(pkg_root / 'configs' / f'{pretrained}.yaml')
            self.ckpt_path = str(pkg_root / 'pretrained' / f'{pretrained}.pth')
            assert os.path.isfile(self.cfg_path), f"Config not found: {self.cfg_path}"
            assert os.path.isfile(self.ckpt_path), f"Checkpoint not found: {self.ckpt_path}"
            self.cfg = OmegaConf.load(self.cfg_path)
        # Manual path specification
        else:
            assert cfg_path is not None and ckpt_path is not None
            self.cfg_path = cfg_path
            self.ckpt_path = ckpt_path
            assert os.path.isfile(self.cfg_path)
            assert os.path.isfile(self.ckpt_path)
            self.cfg = OmegaConf.load(self.cfg_path)

        ##### BACKBONE #####
        self.patch = self.cfg.backbone.patch
        self.diff_backbone = (self.cfg.backbone.diff_pipe_name is not None)
        if self.diff_backbone: # hugging-face diffusion pipeline backbones
            from uplift.extractors.diff_extractor import DiffExtractor
            self.extractor_type = 'diff'
            self.backbone_name = self.cfg.backbone.diff_pipe_name.split('/')[-1] + '_vae'
            self.extractor = DiffExtractor(self.cfg.backbone.diff_pipe_name, self.device)
        else: # ViT extractor
            from uplift.extractors.vit_wrapper import PretrainedViTWrapper
            self.extractor_type = 'vit'
            self.backbone_name = self.cfg.backbone.model_type
            self.extractor = PretrainedViTWrapper(self.cfg.backbone.model_type).to(self.device)
        if not self.silent:
            print(f"Using extractor type: {self.extractor_type}, with backbone: {self.backbone_name}")

        ##### PREPROC #####
        if not self.no_transform:
            # no_UEP = no UPLiFT Encoder Preprocessing
            self.no_UEP = getattr(self.cfg.uplift, 'no_UEP', False)
            # transforms
            if self.diff_backbone: # no normalization for diff backbones
                self.transform = transform=transforms.Compose([transforms.v2.RGB(), transforms.ToTensor()])
            else: # normalization for other backbones
                mean = (0.485, 0.456, 0.406) if "dino" in self.cfg.backbone.model_type else (0.5, 0.5, 0.5)
                std = (0.229, 0.224, 0.225) if "dino" in self.cfg.backbone.model_type else (0.5, 0.5, 0.5)
                self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

        ##### UPLiFT #####
        self.uplift = UPLiFT(self.cfg.backbone.channel, self.cfg.backbone.patch, self.cfg.uplift, low_mem=self.low_mem)
        if self.ckpt_path is not None:
            if self.ckpt_path.endswith('.safetensors'):
                from safetensors.torch import load_file
                loaded_state_dict = load_file(self.ckpt_path)
            else:
                loaded_state_dict = torch.load(self.ckpt_path, weights_only=True)
            self.uplift.load_state_dict(loaded_state_dict)
        self.uplift.enable_iterative(iters)
        self.uplift.return_steps(self.return_steps)
        if not self.silent:
            pc = sum(p.numel() for p in self.uplift.parameters())
            print('loaded UPLiFT model with config: ' + self.cfg_path)
            print('loaded pretrained checkpoint:' + self.ckpt_path)
            print('UPLiFT params: ' + str(pc))
            print('UPLiFT iters: ' + str(iters))
        self.uplift = self.uplift.to(self.device)
        self.uplift.eval()


    '''
    Expects a PIL.Image Object for input img
    '''
    def forward(self, img, report_latency=False, latency_format="dict"):
        if self.auto_resize:
            img = self.resize_input(img)

        ##### TIMING #####
        tik_preproc, tok_preproc = torch.cuda.Event(True), torch.cuda.Event(True)
        tik_extractor, tok_extractor = torch.cuda.Event(True), torch.cuda.Event(True)
        tik_uplift, tok_uplift = torch.cuda.Event(True), torch.cuda.Event(True)

        ##### PREPROC #####
        if not self.no_transform:
            if self.extractor_type == 'diff' and not self.no_UEP:
                tik_preproc.record()
                image_in = extractor.preproc(img).to(self.device)
                tok_preproc.record()
            else:
                tik_preproc.record()
                image_in = self.transform(img)
                if isinstance(image_in, list):
                    image_in = torch.stack(image_in, dim=0)
                image_in = image_in.to(self.device)
                tok_preproc.record()
                if len(image_in.shape) == 3: # we expect [B,C,H,W] input
                    image_in = torch.unsqueeze(image_in, 0)
        else:
            image_in = img

        ##### BACKBONE FEATURES #####
        with torch.no_grad():
            if self.extractor_type == 'diff':
                tik_extractor.record()
                feat = self.extractor(img) # Diffuser backbone accepts raw PIL.Image as input
                tok_extractor.record()
            else:
                tik_extractor.record()
                feat = self.extractor(image_in)[0]
                tok_extractor.record()

        ##### UPLiFT #####
        base_feat = feat
        with torch.no_grad():
            tik_uplift.record()
            feat = self.uplift(image_in, feat, self.out_size)
            tok_uplift.record()

        ##### RETURN #####
        if report_latency: # (optional) return with latency info
            torch.cuda.synchronize()
            batch_size = img.shape[0] if isinstance(img, torch.Tensor) else len(img)
            preproc_time = tik_preproc.elapsed_time(tok_preproc) / batch_size if batch_size > 0 else np.nan
            extractor_time = tik_extractor.elapsed_time(tok_extractor) / batch_size if batch_size > 0 else np.nan
            uplift_time = tik_uplift.elapsed_time(tok_uplift) / batch_size if batch_size > 0 else np.nan
            if latency_format == "dict":
                latencies = {
                        'preproc_time_ms': preproc_time,
                        'extractor_time_ms': extractor_time,
                        'uplift_time_ms': uplift_time,
                }
            elif latency_format == "list":
                latencies = [preproc_time, extractor_time, uplift_time]
            if self.return_base_feat:
                return feat, base_feat, latencies
            else:
                return feat, latencies
        if self.return_base_feat: # (optional) return with base features
            return feat, base_feat
        return feat


    # for use with DiffExtractor backbones only, run the decoder to generate an upsampled image
    def decode(self, lat):
        if self.extractor_type != 'diff':
            print('WARNING, UPLiFTExtractor decode can only be used with a DiffExtractor backbone')
            return None
        img = self.extractor.decode(lat, output_type="pil")
        return img


    ###### UTILS ######


    '''
    Expects a PIL.Image object for input img, resizes input image to be a multiple of the patch size, if needed
    '''
    def resize_input(self, img):
        d0, d1 = img.size
        d0r = round(d0/self.patch) * self.patch
        d1r = round(d1/self.patch) * self.patch
        if d0 != d0r or d1 != d1r:
            if not self.silent:
                print('WARNING: UPLiFTExtractor auto_resize enabled, reshaping input to a multiple of patch size: %i'%self.patch)
                print('(%i, %i) -> (%i, %i)'%(d0,d1,d0r,d1r))
            img = img.resize((d0r, d1r), resample=Image.BILINEAR)
        return img


    def compile(
        self,
        backend: str,
        mode: str,
        fullgraph: bool,
        dynamic: bool,
        device: str | torch.device = "cuda",
        set_tf32: bool = True,
        warmup: Callable | None = None,
    ) -> nn.Module:
        """
        Fully compile UPLiFT for inference in place.

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
            Self, which wraps the compiled module.
        """
        self.uplift = self.uplift.compiled(
                device=device,
                backend=backend,
                mode=mode,
                fullgraph=fullgraph,
                dynamic=dynamic,
                set_tf32=set_tf32,
                warmup=warmup
        )
        if self.extractor_type == 'diff':
            assert isinstance(self.extractor, DiffExtractor)
            self.extractor = self.extractor.compiled(
                backend=backend,
                mode=mode,
                fullgraph=fullgraph,
                dynamic=dynamic,
            )
        return self


    def _load_uplift_checkpoint(
        self,
        ckpt_path: Path | str,
        strict: bool = True,
        log_prefix: str = "",
    ):
        ckpt_path = Path(ckpt_path)
        raw = torch.load(ckpt_path, map_location="cpu")
        state = self._unwrap_state_dict(raw)
        state = self._strip_prefixes(state, prefixes=("module.", "model.", "uplift."))
        incompat = self.uplift.load_state_dict(state, strict=strict)

        if log_prefix and (incompat.missing_keys or incompat.unexpected_keys):
            mk = len(incompat.missing_keys)
            uk = len(incompat.unexpected_keys)
            head_m = incompat.missing_keys[:8]
            head_u = incompat.unexpected_keys[:8]
            if mk:
                print(
                    f"{log_prefix}[warn] Missing keys ({mk}): {head_m}{' ...' if mk>8 else ''}"
                )
            if uk:
                print(
                    f"{log_prefix}[warn] Unexpected keys ({uk}): {head_u}{' ...' if uk>8 else ''}"
                )
        return incompat


    def _unwrap_state_dict(self, raw):
        if not isinstance(raw, dict):
            # assume it's already a flat state_dict
            return raw
        # prefer EMA if available
        for k in ("state_dict_ema", "ema_state_dict", "model_ema", "ema"):
            if k in raw and isinstance(raw[k], dict):
                return raw[k]
        for k in ("state_dict", "model", "net", "uplift", "weights"):
            if k in raw and isinstance(raw[k], dict):
                return raw[k]
        # if it "looks like" a flat state_dict (all tensors), return as-is
        if all(isinstance(v, torch.Tensor) for v in raw.values()):
            return raw
        raise ValueError(
            "Checkpoint does not contain a recognizable state_dict. "
            f"Top-level keys: {list(raw.keys())[:10]}"
        )


    def _strip_prefixes(self, state, prefixes=("module.", "model.", "uplift.")):
        """Remove any of the given prefixes from state_dict keys."""
        new_state = {}
        for k, v in state.items():
            for p in prefixes:
                if k.startswith(p):
                    k = k[len(p) :]
                    break
            new_state[k] = v
        return new_state
