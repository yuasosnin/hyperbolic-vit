import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from oml.interfaces.distances import IDistance
from src.hyptorch import pmath


class HyperbolicDistance(IDistance):
    def __init__(self, c: float, train_c: bool = True):
        super().__init__()
        self.c = nn.Parameter(torch.tensor([float(c)]), requires_grad=train_c)

    def elementwise(self, x1: Tensor, x2: Tensor) -> Tensor:
        return pmath.dist(x1, x2, c=self.c)

    def pairwise(self, x1: Tensor, x2: Tensor) -> Tensor:
        return pmath.dist_matrix(x1, x2, c=self.c)


@torch.jit.script
def _spherical_distance(dot_product: Tensor, r: float = 1.0) -> Tensor:
    return r * torch.arccos(dot_product / (r**2))


class SphericalDistance(IDistance):
    def elementwise(self, x1: Tensor, x2: Tensor) -> Tensor:
        dot_product = (x1 * x2).sum(-1, keepdim=True)
        return _spherical_distance(dot_product)
    
    def pairwise(self, x1: Tensor, x2: Tensor) -> Tensor:
        dot_product = x1 @ x2.transpose(0, 1)
        return _spherical_distance(dot_product)
