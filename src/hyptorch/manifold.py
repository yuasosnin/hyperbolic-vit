from scipy.special import beta

import torch
from torch import Tensor
from geoopt.manifolds import PoincareBall as _PoincareBall
import geoopt.manifolds.stereographic.math as pmath

import src.hyptorch.pmath as pmath2


class PoincareBall(_PoincareBall):
    def beta_split(self, x, n: int):
        # https://github.com/mil-tokyo/hyperbolic_nn_plusplus/blob/main/geoopt_plusplus/modules/linear.py
        """
        Splits last dimension into n chunks: [-1, d] -> [-1, n, d / n]
        """
        dims = x.shape[:-1]
        d = x.shape[-1]
        assert d % n == 0, "n does not divide last dim size"

        betas = beta((d/n)/2, 1/2) / beta(d/2, 1/2)
        x_e = self.logmap0(x).reshape(*dims, n, d // n)  # -1, N, D / N
        return self.expmap0(betas * x_e)

    def beta_concat(self, x):
        """
        Concatenates last 2 dimensions: [-1, n, d / n] -> [-1, d]
        """
        dims = x.shape[:-2]
        n, dn = x.shape[-2:]
        d = n * dn

        betas = beta(d/2, 1/2) / beta((d/n)/2, 1/2)
        x_e = self.logmap0(x).reshape(*dims, d)  # -1, D
        return self.expmap0(betas * x_e)
    
    # def mobius_add(self, x: Tensor, y: Tensor, *, dim=-1, project=True) -> Tensor:
    #     res = pmath2.mobius_add(x, y, k=self.k)
    #     if project:
    #         return pmath.project(res, k=self.k)
    #     else:
    #         return res

    # def mobius_sub(self, x: Tensor, y: Tensor, *, dim=-1, project=True) -> Tensor:
    #     res = pmath2.mobius_add(x, -y, k=self.k)
    #     if project:
    #         return pmath.project(res, k=self.k)
    #     else:
    #         return res
    
    def weighted_midpoint(self, xs, weights, *, posweight=False, project=True):
        mid = pmath2.weighted_midpoint(xs, self.k, weights, posweight=posweight)
        if project:
            return pmath.project(mid, k=self.k, dim=-1)
        else:
            return mid
    
    def dist(self, x, y, dim=-1, keepdim: bool = False):
        return 2 / torch.sqrt(-self.k) * torch.arctanh(
            torch.sqrt(-self.k) * pmath2._mobius_add_norm(-x, y, self.k))
