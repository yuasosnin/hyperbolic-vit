import torch
import torch.nn as nn
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
    