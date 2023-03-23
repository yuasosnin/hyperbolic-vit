import torch
from .hyptorch import pmath


def hyper_elementwise_dist(x1, x2, c):
    # [n, k], [n, k] -> [n, 1]
    bs = x1.shape[0]
    dist = torch.empty(bs, device=x1.device)
    for i in range(bs):
        dist[i] = pmath.dist(x1[i], x2[i], c=c, keepdim=False)
    return dist

def hyper_pairwise_dist(x1, x2, c):
    # [n, k], [m, k] -> [n, m]
    return pmath.dist_matrix(x1, x2, c=c)
