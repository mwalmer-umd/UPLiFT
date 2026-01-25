"""
Feature Cache Manager to accelerate training. Feature Caching is recommended for models with low
latent channel depth (like Diffusers VAE) but not for models with large latent channel depth
(like DINOv2 and DINOv3) which leads to very large cache sizes.

Code by: Matthew Walmer
"""
import os
import torch
import numpy as np


class CacheManager():
    def __init__(self, cache_dir, dataset, dataroot, extractor_name, device):
        self.cache_dir = cache_dir
        self.dataset = dataset
        self.dataroot = dataroot
        self.extractor_name = extractor_name
        self.device = device
        self.version = 'v2' # version of the Cache Manager
        self.full_dir = os.path.join(cache_dir, self.version, self.dataset, self.extractor_name)

    def get_cache_loc(self, filename, res):
        f_end = filename.replace(self.dataroot,'')
        f_cache = os.path.join(self.full_dir, str(res), f_end+'.npy')
        return f_cache

    # load cache files for filenames at resolution res
    # returns None if any files missing from the cache
    def load_cache(self, filenames, res):
        feats = []
        for i, f in enumerate(filenames):
            f_cache = self.get_cache_loc(f, res)
            if not os.path.isfile(f_cache):
                return None
            f_data = np.load(f_cache)
            feats.append(torch.from_numpy(f_data))
        feats = torch.stack(feats).to(self.device)
        return feats

    def save_cache(self, filenames, feat, res):
        feat = feat.cpu().numpy()
        for i, f in enumerate(filenames):
            f_cache = self.get_cache_loc(f, res)
            f_dir = os.path.dirname(f_cache)
            os.makedirs(f_dir, exist_ok=True)
            np.save(f_cache, feat[i,...])