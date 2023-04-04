import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from oml.interfaces.distances import IDistance
from geoopt import Manifold


class ManifoldDistance(IDistance):
    def __init__(self, manifold: Manifold):
        super().__init__()
        self.manifold = manifold

    def elementwise(self, x1: Tensor, x2: Tensor) -> Tensor:
        return self.manifold.dist(x1, x2)

class ManifoldProjection(nn.Module):
    def __init__(self, manifold: Manifold):
        super().__init__()
        self.manifold = manifold

    def forward(self, x):
        return self.manifold.projx(self.manifold.expmap0(x))
