import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from oml.interfaces.distances import IDistance
from geoopt import Manifold


class ManifoldDistance(IDistance):
    def __init__(self, manifold: Manifold):
        self.manifold = manifold

    def elementwise(self, x1: Tensor, x2: Tensor) -> Tensor:
        return self.manifold.dist(x1, x2)
    
    def pairwise(self, x1: Tensor, x2: Tensor) -> Tensor:
        n = x1.shape[0]
        m = x2.shape[0]
        inner = torch.empty((n, m), device=x1.device)
        for i in range(n):
            x1_i = x1[i].unsqueeze(0)
            inner[i, :] = self.manifold.dist(x1_i, x2)
        return inner


class ManifoldProjection(nn.Module):
    def __init__(self, manifold: Manifold):
        self.manifold = manifold

    def forward(self, x):
        return self.manifold.projx(self.manifold.expmap(x))
