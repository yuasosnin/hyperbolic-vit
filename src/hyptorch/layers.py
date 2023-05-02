import torch
import torch.nn as nn

from geoopt import ManifoldParameter
from src.hyptorch.manifold import PoincareBall


class HyperbolicActivation(nn.Module):
    def __init__(self, fn, ball=None):
        super().__init__()
        self.ball = ball or PoincareBall(c=1.0)
        self.fn = fn

    def forward(self, x):
        x_e = self.ball.logmap0(x)
        x_e = self.fn(x_e)
        return self.ball.expmap0(x_e)


class HyperbolicLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, ball=None, **kwargs):
        super().__init__(in_features, out_features, bias=bias, **kwargs)
        self.ball = ball or PoincareBall(c=1.0)
        if self.bias is not None:
            self.bias = ManifoldParameter(self.bias, manifold=self.ball)

    def forward(self, input):
        output = self.ball.mobius_matvec(self.weight, input)
        if self.bias is not None:
            output = self.ball.mobius_add(output, self.bias)
        return output


# class HyperbolicLinearPP(nn.Module):
#     """https://github.com/mil-tokyo/hyperbolic_nn_plusplus"""
#     def __init__(self, in_features, out_features, bias=True, ball=None):
#         super().__init__()
#         self.ball = ball or PoincareBall(c=1.0)
#         weight = torch.empty(in_features, out_features).normal_(
#             mean=0, std=(2 * in_features * out_features) ** -0.5)
#         self.weight_g = nn.Parameter(weight.norm(dim=0))
#         self.weight_v = nn.Parameter(weight)
#         self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=bias)

#     def forward(self, x):
#         return self.ball.poincare_linear(
#             x,
#             self.weight_g,
#             self.weight_v / self.weight_v.norm(dim=0).clamp_min(1e-15),
#             self.bias)


class HyperbolicDistanceAttention(nn.Module):
    def __init__(self, dim, num_heads=8, ball=None, qkv_bias=False, qk_scale=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.ball = ball or PoincareBall(c=1.0)
        self.num_heads = num_heads
        # Learned beta parameter is sometimes used for scaling
        self.scale = nn.Parameter(torch.tensor(1.0), requires_grad=qk_scale)

        # Contrary to default implementation, we have separate q, k, v
        # and manually adjust pretrained weights to match the keys.
        self.q = HyperbolicLinear(dim, dim, bias=qkv_bias)
        self.k = HyperbolicLinear(dim, dim, bias=qkv_bias)
        self.v = HyperbolicLinear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = HyperbolicLinear(dim, dim)
        # Not sure about dropout in hyperbolic space
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, L, D = x.shape
        qkv = torch.stack([self.q(x), self.k(x), self.v(x)], dim=-2)  # B, L, 3, D
        qkv = self.ball.beta_split(qkv, n=self.num_heads)  # B, L, 3, H, D/H
        qkv = qkv.permute(2, 0, 3, 1, 4)  # 3, B, H, L, D/H
        q, k, v = qkv.unbind(0)  # B, H, L, D/H

        attn = torch.softmax(self.scale * -self.ball.dist(q.unsqueeze(-2), k.unsqueeze(-3)), dim=-1)  # B, H, L, L
        attn = self.attn_drop(attn)
        y = self.ball.weighted_midpoint(v, attn)  # B, H, L, D/H
        y = self.ball.beta_concat(y.transpose(1, 2))  # B, L, D
        y = self.proj_drop(self.proj(y))
        return y, attn


class ExponentialMap(nn.Module):
    def __init__(self, manifold=None):
        super().__init__()
        self.ball = manifold or PoincareBall(c=1.0)

    def forward(self, x, u=None):
        if u is not None:
            return self.ball.expmap(x, u)
        return self.ball.expmap0(x)


class LogarithmicMap(nn.Module):
    def __init__(self, manifold=None):
        super().__init__()
        self.ball = manifold or PoincareBall(c=1.0)

    def forward(self, x, u=None):
        if u is not None:
            return self.ball.logmap(x, u)
        return self.ball.logmap0(x)
