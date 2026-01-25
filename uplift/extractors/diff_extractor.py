"""
Feature extractor for wrapping models from the Hugging Face Diffusers library.

Will load the specified pipeline, extract the VAE, and unload the rest to reduce memory usage.

Code by: Matthew Walmer and Anirud Aggarwal
"""
import time
import gc
from typing import Callable

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from diffusers import DiffusionPipeline


########## UTILS ##########


# from: https://gist.github.com/sayakpaul/3ae0f847001d342af27018a96f467e4e
def flush():
    gc.collect()
    torch.cuda.empty_cache()

# from: https://gist.github.com/sayakpaul/3ae0f847001d342af27018a96f467e4e
def bytes_to_giga_bytes(bytes):
    return bytes / 1024 / 1024 / 1024

# from: https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_img2img.py
def retrieve_latents(encoder_output, generator=None, sample_mode="sample"):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


########## EXTRACTOR ##########


class DiffExtractor():
    def __init__(self, pipe_name, device, dtype=torch.float32):
        self.pipe_name = pipe_name
        self.device = device
        # load pipeline
        pipeline = DiffusionPipeline.from_pretrained(
            self.pipe_name,
            text_encoder=None,
            torch_dtype=dtype).to(self.device)
        self.vae = pipeline.vae
        self.image_processor = pipeline.image_processor
        # delete other elements
        del pipeline
        flush()


    def preproc(self, img):
        img = self.image_processor.preprocess(img).to(device=self.device, dtype=self.vae.dtype)
        return img


    def postproc(self, img, output_type="pil"):
        img = self.image_processor.postprocess(img, output_type=output_type)
        if output_type == "pil":
            img = img[0]
        return img


    @torch.no_grad()
    def encode(self, img):
        img = self.preproc(img)
        lat = self.vae.encode(img)
        lat = retrieve_latents(lat, sample_mode="argmax")
        return lat


    @torch.no_grad()
    def decode(self, lat, output_type="pil"):
        lat = lat.to(device=self.device, dtype=self.vae.dtype)
        img = self.vae.decode(lat, return_dict=False)[0]
        img = self.postproc(img, output_type=output_type)
        return img


    def __call__(self, img):
        return self.encode(img)


    def compiled(
        self,
        backend: str = "inductor",
        mode: str = "max-autotune",
        fullgraph: bool = True,
        dynamic: bool = False,
    ) -> "DiffExtractor":
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.vae: nn.Module = torch.compile(  # type: ignore
            self.vae,
            backend=backend,
            mode=mode,
            fullgraph=fullgraph,
            dynamic=dynamic,
        )

        return self