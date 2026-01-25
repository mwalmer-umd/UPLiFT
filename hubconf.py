dependencies = ["torch", "huggingface_hub", "omegaconf", "safetensors"]


def uplift(variant="dinov2-s14", pretrained=True, include_extractor=True, **kwargs):
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
    from uplift.hub_loader import load_model
    return load_model(variant, pretrained, include_extractor, **kwargs)


def uplift_dinov2_s14(pretrained=True, include_extractor=True, **kwargs):
    """Load UPLiFT with DINOv2-S/14 backbone."""
    return uplift("dinov2-s14", pretrained, include_extractor, **kwargs)


def uplift_dinov3_splus16(pretrained=True, include_extractor=True, **kwargs):
    """Load UPLiFT with DINOv3-S+/16 backbone."""
    return uplift("dinov3-splus16", pretrained, include_extractor, **kwargs)


def uplift_sd15_vae(pretrained=True, include_extractor=True, **kwargs):
    """Load UPLiFT with Stable Diffusion 1.5 VAE backbone."""
    return uplift("sd15-vae", pretrained, include_extractor, **kwargs)
