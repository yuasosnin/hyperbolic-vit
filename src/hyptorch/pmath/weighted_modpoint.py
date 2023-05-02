from typing import Optional

import torch
from torch import Tensor
import geoopt.manifolds.stereographic.math as pmath


def weighted_midpoint(
        xs: Tensor,
        k: Tensor,
        weights: Optional[Tensor] = None,
        posweight: bool = False,
    ):
    """
    Possible shapes:
    xs: [..., N, D]
    N - dim to compute midpoint over
    D - main vector dim
    weights: [..., N] OR [..., M, N]
    M - compute midpoints over M sets of weights in parallel (for attention)
    `...` (batch, multihead, etc) dims must match too.
    output: [..., D] OR [..., M, D]
    """
    squeeze_flag = False
    if weights is None:
        weights = torch.tensor(1.0, dtype=xs.dtype, device=xs.device)
    elif len(weights.shape) + 1 == len(xs.shape):
        # single weights verison proveded, augment missing dim
        squeeze_flag = True
        weights = weights.unsqueeze(-2)
    
    gamma = pmath.lambda_x(xs, k=k, dim=-1, keepdim=True)
    if posweight and weights.lt(0).any():
        xs = torch.where(weights.lt(0), pmath.antipode(xs, k=k, dim=-1), xs)
        weights = weights.abs()
    denominator = weights @ (gamma - 1)
    nominator = weights @ (gamma * xs)
    two_mean = nominator / denominator.clamp_min(1e-15)
    a_mean = pmath.mobius_scalar_mul(
        torch.tensor(0.5, dtype=xs.dtype, device=xs.device), two_mean, k=k, dim=-1)
    return a_mean.squeeze(-2) if squeeze_flag else a_mean
