import torch

from oml.models.vit.vit import ViTExtractor
from oml.models.vit.hubconf import (
    dino_vitb8,
    dino_vitb16,
    dino_vits8,
    dino_vits16,
)

from src.vision_transformer_hyperbolic import hvit_base, hvit_small


def dino_hvits16(pretrained=True, **kwargs):
    """
    ViT-Small/16x16 pre-trained with DINO.
    Achieves 74.5% top-1 accuracy on ImageNet with k-NN classification.
    """
    model = hvit_small(patch_size=16, num_classes=0, **kwargs)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth",
            map_location="cpu",
        )
        model.load_state_dict(state_dict, strict=True)
    return model


def dino_hvits8(pretrained=True, **kwargs):
    """
    ViT-Small/8x8 pre-trained with DINO.
    Achieves 78.3% top-1 accuracy on ImageNet with k-NN classification.
    """
    model = hvit_small(patch_size=8, num_classes=0, **kwargs)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth",
            map_location="cpu",
        )
        model.load_state_dict(state_dict, strict=True)
    return model


def dino_hvitb16(pretrained=True, **kwargs):
    """
    ViT-Base/16x16 pre-trained with DINO.
    Achieves 76.1% top-1 accuracy on ImageNet with k-NN classification.
    """
    model = hvit_base(patch_size=16, num_classes=0, **kwargs)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth",
            map_location="cpu",
        )
        model.load_state_dict(state_dict, strict=True)
    return model


def dino_hvitb8(pretrained=True, **kwargs):
    """
    ViT-Base/8x8 pre-trained with DINO.
    Achieves 77.4% top-1 accuracy on ImageNet with k-NN classification.
    """
    model = hvit_base(patch_size=8, num_classes=0, **kwargs)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth",
            map_location="cpu",
        )
        model.load_state_dict(state_dict, strict=True)
    return model


class HViTExtractor(ViTExtractor):
    constructors = {
        "vits8": dino_vits8, 
        "vits16": dino_vits16, 
        "vitb8": dino_vitb8, 
        "vitb16": dino_vitb16,
        "hvits8": dino_hvits8, 
        "hvits16": dino_hvits16, 
        "hvitb8": dino_hvitb8, 
        "hvitb16": dino_hvitb16
        }
