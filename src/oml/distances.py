import torch
from torch import Tensor

from oml.interfaces.distances import IDistance
from src.hyptorch import PoincareBall
from geoopt import Manifold


class ManifoldDistance(IDistance):
    def __init__(self, manifold: Manifold):
        super().__init__()
        self.manifold = manifold

    def elementwise(self, x1: Tensor, x2: Tensor) -> Tensor:
        return self.manifold.dist(x1, x2)

    def pairwise(self, x1: Tensor, x2: Tensor) -> Tensor:
        return self.manifold.dist(x1.unsqueeze(-2), x2.unsqueeze(-3))


class PoincareBallDistance(IDistance):
    def __init__(self, manifold=None):
        super().__init__()
        self.ball = manifold or PoincareBall(c=1.0)

    def elementwise(self, x1: Tensor, x2: Tensor) -> Tensor:
        return self.ball.dist(x1, x2)

    def pairwise(self, x1: Tensor, x2: Tensor) -> Tensor:
        return self.ball.dist(x1.unsqueeze(-2), x2.unsqueeze(-3))


class DotProductDistance(IDistance):
    def elementwise(self, x1: Tensor, x2: Tensor) -> Tensor:
        return -(x1 * x2).sum(-1)

    def pairwise(self, x1: Tensor, x2: Tensor) -> Tensor:
        return -(x1 @ x2.transpose(-2, -1))
