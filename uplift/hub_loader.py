"""
Code to help load pretrained UPLiFT models and (optional) corresponding
backbones for easy feature extraction and upsampling.

Code by: Anirud Aggarwal
"""
from pathlib import Path
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf

HF_ORG = "UPLiFT-upsampler"
RELEASE_SUFFIX = ""

VARIANTS = {
    "dinov2-s14": {
        "hf_repo": "uplift_dinov2-s14",
        "hf_filename": "uplift_dinov2-s14.safetensors",
        "config": "uplift_dinov2-s14.yaml",
        "extra": "vit",
        "check_package": "timm",
    },
    "dinov3-splus16": {
        "hf_repo": "uplift_dinov3-splus16",
        "hf_filename": "uplift_dinov3-splus16.safetensors",
        "config": "uplift_dinov3-splus16.yaml",
        "extra": "vit",
        "check_package": "timm",
    },
    "sd15-vae": {
        "hf_repo": "uplift_sd1.5vae",
        "hf_filename": "uplift_sd1.5vae.safetensors",
        "config": "uplift_sd1.5vae.yaml",
        "extra": "sd-vae",
        "check_package": "diffusers",
    },
}


def _check_dependencies(variant: str):
    info = VARIANTS[variant]
    try:
        __import__(info["check_package"])
    except ImportError:
        raise ImportError(
            f"'{variant}' requires '{info['check_package']}'. "
            f"Install with: pip install uplift[{info['extra']}]"
        )


def _get_config_path(variant: str) -> Path:
    return Path(__file__).parent / "configs" / VARIANTS[variant]["config"]


def _download_weights(variant: str, cache_dir: str = None) -> str:
    info = VARIANTS[variant]
    repo_id = f"{HF_ORG}/{info['hf_repo']}{RELEASE_SUFFIX}"
    return hf_hub_download(
        repo_id=repo_id,
        filename=info["hf_filename"],
        cache_dir=cache_dir,
    )


def load_model(variant: str, pretrained: bool = True,
               include_extractor: bool = True, **kwargs):
    """
    Load UPLiFT model.

    Args:
        variant: Model variant - "dinov2-s14", "dinov3-splus16", or "sd15-vae"
        pretrained: Load pretrained weights from HuggingFace Hub
        include_extractor: If True, return UPLiFTExtractor with backbone included.
                          If False, return raw UPLiFT model only.
        **kwargs: Passed to UPLiFTExtractor (if include_extractor=True):
            - iters (int): Upsampling iterations (default: 4)
            - return_base_feat (bool): Return backbone features too (default: False)
            - silent (bool): Disable print statements (default: False)

    Returns:
        UPLiFTExtractor or UPLiFT model
    """
    if variant not in VARIANTS:
        raise ValueError(
            f"Unknown variant '{variant}'. Choose from: {list(VARIANTS.keys())}"
        )

    if include_extractor:
        _check_dependencies(variant)

    config_path = _get_config_path(variant)
    weights_path = _download_weights(variant) if pretrained else None

    if include_extractor:
        from uplift.uplift_extractor import UPLiFTExtractor
        return UPLiFTExtractor(
            config=str(config_path),
            weights=weights_path,
            **kwargs
        )
    else:
        from uplift.uplift import UPLiFT
        cfg = OmegaConf.load(config_path)
        model = UPLiFT(cfg.backbone.channel, cfg.backbone.patch, cfg.uplift)
        if weights_path:
            from safetensors.torch import load_file
            state_dict = load_file(weights_path)
            model.load_state_dict(state_dict)
        return model
