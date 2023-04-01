import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


class Normalize(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return F.normalize(x, p=2, dim=-1)
