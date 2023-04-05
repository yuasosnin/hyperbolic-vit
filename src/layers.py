from functools import partial

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from geoopt.manifolds.lorentz import math as hmath
from src.hyptorch import pmath


class Normalize(nn.Module):
    def __init__(self, denom_grad=True):
        super().__init__()
        self.denom_grad = denom_grad

    def forward(self, x: Tensor) -> Tensor:
        norm = x.norm(p=2, dim=-1, keepdim=True)
        if self.denom_grad:
            norm = norm.detach()
        return x / norm


def clip_eucledian(x, r=1.0):
    x_norm = x.norm(p=2, dim=-1, keepdim=True) + 1e-5
    coefficient = torch.minimum(torch.ones_like(x_norm), r / x_norm)
    return x * coefficient


class PoincareBallProjection(nn.Module):
    def __init__(self, c, clip_r=1.0):
        super().__init__()
        self.c = c
        self.clip_r = clip_r

    def forward(self, x):
        if self.clip_r is not None:
            x = clip_eucledian(x, self.clip_r)
        out = pmath.project(pmath.expmap0(x, c=self.c), c=self.c)
        out.register_hook(partial(self._riemannian_gradient_scale, t=out))
        return out
    
    def _riemannian_gradient_scale(self, grad, t) -> Tensor:
        scale = (1 - self.c * t.pow(2).sum(-1, keepdim=True)).pow(2) / 4
        return grad * scale


class LorentzProjection(nn.Module):
    def __init__(self, c: float):
        super().__init__()
        self.c = c

    def forward(self, x):
        return hmath.expmap0(x, k=self.c)
