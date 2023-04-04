import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from geoopt.manifolds.lorentz import math as hmath
from src.hyptorch.nn import ToPoincare as PoincareBallProjection


class Normalize(nn.Module):
    def __init__(self, denom_grad=True):
        super().__init__()
        self.denom_grad = denom_grad

    def forward(self, x: Tensor) -> Tensor:
        norm = x.norm(p=2, dim=-1, keepdim=True)
        if self.denom_grad:
            norm = norm.detach()
        return x / norm


class LorentzProjection(nn.Module):
    def __init__(self, c: float, train_c: bool = False):
        self.c = nn.Parameter(torch.tensor([float(c)]), requires_grad=train_c)

    def forward(self, x):
        return hmath.project(hmath.expmap0(x, k=self.c), k=self.c)
