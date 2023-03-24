import torch
import torch.nn as nn
from torch import Tensor

from oml.interfaces.distances import IDistance
from .hyptorch import pmath


class HyperbolicDistance(IDistance):
    def __init__(self, c, train_c: bool = True):
        self.c = nn.Parameter(data=[c], requires_grad=train_c)

    def elementwise(self, x1: Tensor, x2: Tensor) -> Tensor:
        bs = x1.shape[0]
        dist = torch.empty(bs, device=x1.device)
        for i in range(bs):
            dist[i] = pmath.dist(x1[i], x2[i], c=self.c, keepdim=False)
        return dist

    def pairwise(self, x1: Tensor, x2: Tensor) -> Tensor:
        return pmath.dist_matrix(x1, x2, c=self.c)
    