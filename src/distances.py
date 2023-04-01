import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from oml.interfaces.distances import IDistance
from .hyptorch import pmath


class HyperbolicDistance(IDistance, nn.Module):
    def __init__(self, c, train_c: bool = True):
        super().__init__()
        self.c = nn.Parameter(torch.tensor([float(c)]), requires_grad=train_c)

    def elementwise(self, x1: Tensor, x2: Tensor) -> Tensor:
        return pmath.dist(x1, x2, c=self.c)

    def pairwise(self, x1: Tensor, x2: Tensor) -> Tensor:
        return pmath.dist_matrix(x1, x2, c=self.c)


@torch.jit.script
def _sphere_distance(dot_product: Tensor, r: float = 1.0):
    return r * torch.arccos(dot_product / (r**2))


def to_unit_sphere(x):
    x = F.normalize(x, p=2, dim=-1)


class SphericalDistance(IDistance):
    def elementwise(self, x1: Tensor, x2: Tensor) -> Tensor:
        dot_product = (x1 * x2).sum(-1, keepdim=True)
        return _sphere_distance(dot_product)
    
    def pairwise(self, x1: Tensor, x2: Tensor) -> Tensor:
        dot_product = x1 @ torch.transpose(x2, 0, 1)
        return _sphere_distance(dot_product)


class ToSphere(nn.Module):
    def forward(x: Tensor):
        return to_unit_sphere(x)
