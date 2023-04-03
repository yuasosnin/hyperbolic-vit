import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


class Normalize(nn.Module):
    def __init__(self, grad_denom=True):
        super().__init__()
        self.grad_denom = grad_denom

    def forward(self, x: Tensor) -> Tensor:
        norm = x.norm(p=2, dim=-1, keepdim=True)
        if self.grad_denom:
            norm = norm.detach()
        return x / norm
