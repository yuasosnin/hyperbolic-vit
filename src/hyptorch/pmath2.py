from typing import Optional

import torch
from torch import Tensor

from src.hyptorch import pmath


def _outer(x, y):
    return x @ torch.transpose(y, 0, 1)


@torch.jit.script
def mobius_add(x: Tensor, y: Tensor, c: Tensor, eps: float = 1e-5):
    x2 = x.pow(2).sum(dim=-1, keepdim=True)
    y2 = y.pow(2).sum(dim=-1, keepdim=True)
    xy = (x * y).sum(dim=-1, keepdim=True)

    denom = 1 + 2 * c * xy + c**2 * x2 * y2
    alpha = 1 + 2 * c * xy + c * y2
    beta = 1 - c * x2

    return (alpha * x + beta * y) / (denom + eps)


@torch.jit.script
def mobius_add_outer(x: Tensor, y: Tensor, c: Tensor, eps: float = 1e-5):
    x2 = x.pow(2).sum(dim=-1, keepdim=True)  # N x 1
    y2 = y.pow(2).sum(dim=-1, keepdim=True)  # M x 1
    xy = _outer(x, y)  # N x M

    denom = 1 + 2 * c * xy + c**2 * _outer(x2, y2)  # N x M
    alpha = 1 + 2 * c * xy + c * y2.permute(1, 0)  # N x M
    beta = (1 - c * x2).expand(-1, y.shape[0])  # N x M

    return (alpha.unsqueeze(2) * x.unsqueeze(1) + beta.unsqueeze(2) * y.unsqueeze(0)) / (denom.unsqueeze(2) + eps)


@torch.jit.script
def _loop_mobius_add_outer(x, y, c, eps: float = 1e-5):
    n = x.shape[0]
    m = y.shape[0]
    d = x.shape[1]
    inner = torch.empty((n, m, d), device=x.device)
    for i in range(n):
        x_i = x[i].unsqueeze(0)
        inner[i, :] = mobius_add(x_i, y, c, eps=eps)
    return inner


@torch.jit.script
def mobius_add_norm(x: Tensor, y: Tensor, c: Tensor, keepdim: bool = False, eps: float = 1e-5):
    """
    Calculates norm of mobius addition of 2 tensors efficiently.
    Accepts tensors of equal shape, i.e. N x D, N x D -> N,
    where A_i = ||X_i \madd Y_i||^2
    As in Hyperbolic Image Segmentation (2022).
    """
    assert x.shape == y.shape
    x2 = x.pow(2).sum(dim=-1, keepdim=keepdim)
    y2 = y.pow(2).sum(dim=-1, keepdim=keepdim)
    xy = (x * y).sum(dim=-1, keepdim=keepdim)
    
    denom = 1 + 2 * c * xy + c**2 * x2 * y2
    alpha = (1 + 2 * c * xy + c * y2) / (denom + eps)
    beta = (1 - c * x2) / (denom + eps)

    return (alpha**2 * x2) + 2*alpha*beta*xy + (beta**2)*y2


@torch.jit.script
def mobius_add_norm_outer(x: Tensor, y: Tensor, c: Tensor, eps: float = 1e-5):
    """
    Calculates norm of mobius addition of 2 tensors efficiently.
    Does so in an outer manner, i.e. N x D, M x D -> N x M,
    where A_ij = ||X_i \madd Y_j||^2
    """
    x2 = x.pow(2).sum(dim=-1, keepdim=True)  # N x 1
    y2 = y.pow(2).sum(dim=-1, keepdim=True)  # M x 1
    xy = _outer(x, y)  # N x M

    denom = (1 + 2 * c * xy + c**2 * _outer(x2, y2))  # N x M
    alpha = (1 + 2 * c * xy + c * y2.permute(1, 0)) / (denom + eps)  # N x M
    beta = (1 - c * x2).expand(-1, y.shape[0]) / (denom + eps)  # N x M

    return (alpha**2 * x2) + 2 * alpha * beta * xy + (beta**2 * y2.permute(1, 0))


@torch.jit.script
def dist(x: Tensor, y: Tensor, c: Tensor, keepdim: bool = False, eps: float = 1e-5):
    return 2 / torch.sqrt(c) * torch.arctanh(
        torch.sqrt(c) * mobius_add_norm(-x, y, c, keepdim=keepdim, eps=eps))

@torch.jit.script
def dist_matrix(x: Tensor, y: Tensor, c: Tensor, eps: float = 1e-5):
    return 2 / torch.sqrt(c) * torch.arctanh(
        torch.sqrt(c) * mobius_add_norm_outer(-x, y, c, eps=eps))

@torch.jit.script
def _loop_dist_matrix(x, y, c):
    n = x.shape[0]
    m = y.shape[0]
    inner = torch.empty((n, m), device=x.device)
    for i in range(n):
        x_i = x[i].unsqueeze(0)
        inner[i, :] = mobius_add(-x_i, y, c).norm(p=2, dim=-1)
    return 2 / torch.sqrt(c) * torch.arctanh(torch.sqrt(c) * inner)


def exponential_map(x: Tensor, c: Tensor, u: Optional[Tensor] = None, safe: bool = True):
    c = torch.as_tensor(c).type_as(x)
    if u is None:
        x_p = pmath._expmap0(x, c)
    else:
        x_p = pmath._expmap(x, u, c)
    if safe:
        x_p = pmath._project(x_p, c)
    return x_p

def logarithmic_map(x: Tensor, c: Tensor, u: Optional[Tensor] = None, safe: bool = True):
    c = torch.as_tensor(c).type_as(x)
    if u is None:
        x_p = pmath._logmap0(x, c)
    else:
        x_p = pmath._logmap(x, u, c)
    return x_p
