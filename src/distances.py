import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from oml.interfaces.distances import IDistance
from src.hyptorch import pmath
from src.hyptorch import pmath2


class PoincareBallDistance(IDistance):
    def __init__(self, c: float, train_c: bool = True):
        super().__init__()
        self.c = nn.Parameter(torch.tensor([c], dtype=torch.float), requires_grad=train_c)

    def elementwise(self, x1: Tensor, x2: Tensor) -> Tensor:
        return pmath2.dist(x1, x2, c=self.c)
    
    def pairwise(self, x1: Tensor, x2: Tensor) -> Tensor:
        return pmath2.dist_matrix(x1, x2, c=self.c)


class LorentzDistance(IDistance):
    def __init__(self, c: float, train_c: bool = True):
        super().__init__()
        self.c = nn.Parameter(torch.tensor([c], dtype=torch.float), requires_grad=train_c)
    
    @staticmethod
    def _lorentz_distance(dot_product, c):
        return torch.sqrt(c) * torch.arccosh(dot_product / c)

    def elementwise(self, x1: Tensor, x2: Tensor) -> Tensor:
        dot_product = (x1 * x2).sum(-1) - 2 * (x1[:, 0] * x2[:, 0])
        return self._lorentz_distance(dot_product, self.c)
    
    def pairwise(self, x1: Tensor, x2: Tensor) -> Tensor:
        dot_product = x1 @ x2.transpose(0, 1) - 2 * torch.outer(x1[:, 0], x2[:, 0])
        return self._lorentz_distance(dot_product, self.c)


class SphericalDistance(IDistance):
    @staticmethod
    def _spherical_distance(dot_product: Tensor, r: float = 1.0) -> Tensor:
        return r * torch.arccos(dot_product / (r**2))

    def elementwise(self, x1: Tensor, x2: Tensor) -> Tensor:
        dot_product = (x1 * x2).sum(-1)
        return self._spherical_distance(dot_product)
    
    def pairwise(self, x1: Tensor, x2: Tensor) -> Tensor:
        dot_product = x1 @ x2.transpose(0, 1)
        return self._spherical_distance(dot_product)
