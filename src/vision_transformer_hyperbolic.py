from copy import deepcopy

import torch
import torch.nn as nn

from src.hyptorch import PoincareBall, HyperbolicLinear, HyperbolicDistanceAttention, HyperbolicActivation
from src.vision_transformer import DropPath, PatchEmbed, VisionTransformer

# TODO: maybe make pos embedding ManifoldParameter?


class HypMlp(nn.Module):
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            ball=None,
            act_layer=None,
            drop=0.0
        ):
        super().__init__()
        self.ball = ball or PoincareBall(c=1.0)

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = HyperbolicLinear(in_features, hidden_features, ball=self.ball)
        self.act = nn.Identity() if act_layer is None else act_layer()
        self.fc2 = HyperbolicLinear(hidden_features, out_features, ball=self.ball)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class HypBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        ball=None,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.Identity,
        norm_layer=nn.Identity,
    ):
        super().__init__()
        self.ball = ball or PoincareBall(c=1.0)

        self.norm1 = norm_layer(dim)
        self.attn = HyperbolicDistanceAttention(
            dim=dim,
            num_heads=num_heads,
            ball=self.ball,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = HypMlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            ball=self.ball,
            act_layer=act_layer,
            drop=drop)

    def forward(self, x, return_attention: bool = False):
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = self.ball.mobius_add(x, self.drop_path(y))
        return self.ball.mobius_add(x, self.drop_path(self.mlp(self.norm2(x))))


class HyperbolicVisionTransformer(VisionTransformer):
    """Hyperbolic Vision Transformer"""

    def __init__(
            self,
            c=1.0,
            img_size=[224],
            patch_size=16,
            in_chans=3,
            num_classes=0,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.0,
            qkv_bias=False,
            qk_scale=False,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            act_layer=nn.Identity,
            norm_layer=nn.Identity,
            **kwargs
        ):
        super().__init__()
        self.ball = PoincareBall(c=c)

        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(
            img_size=img_size[0], patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        blocks = []
        for i in range(depth):
            block = HypBlock(
                dim=embed_dim,
                num_heads=num_heads,
                ball=self.ball,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                act_layer=act_layer,
                norm_layer=norm_layer)
            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)
        self.norm = norm_layer(embed_dim)

        # Classifier head
        # self.head = HyperbolicMLR(embed_dim, num_classes, ball=self.ball) if num_classes > 0 else nn.Identity()
        self.head = nn.Identity()

        torch.nn.init.trunc_normal_(self.pos_embed, std=0.02)
        torch.nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def forward(self, x):
        # initially map tokens onto poincare ball
        x = self.ball.expmap0(self.prepare_tokens(x))
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]

    def _split_qkv(self, ckpt):
        ckpt = deepcopy(ckpt)
        ckpt_keys = list(ckpt.keys())
        for key in ckpt_keys:
            if key.endswith(".attn.qkv.weight"):
                n = key.split(".")[1]
                weight = ckpt[key]
                q, k, v = weight.reshape(3, self.num_features, self.num_features).unbind(0)
                ckpt[f'blocks.{n}.attn.q.weight'] = q
                ckpt[f'blocks.{n}.attn.k.weight'] = k
                ckpt[f'blocks.{n}.attn.v.weight'] = v
            if key.endswith(".attn.qkv.bias"):
                n = key.split(".")[1]
                bias = ckpt[key]
                q_b, k_b, v_b = bias.reshape(3, self.num_features).unbind(0)
                ckpt[f'blocks.{n}.attn.q.bias'] = q_b
                ckpt[f'blocks.{n}.attn.k.bias'] = k_b
                ckpt[f'blocks.{n}.attn.v.bias'] = v_b
        return ckpt

    def _expmap_bias(self, ckpt):
        ckpt = deepcopy(ckpt)
        ckpt_keys = list(ckpt.keys())
        for key in ckpt_keys:
            is_bias = key.endswith("bias")
            is_patch = key.startswith("patch_embed")
            if is_bias and not is_patch:
                ckpt[key] = self.ball.expmap0(ckpt[key])
        return ckpt

    def load_state_dict(self, state_dict, strict: bool = True):
        state_dict = self._split_qkv(state_dict)
        state_dict = self._expmap_bias(state_dict)
        return super().load_state_dict(state_dict, strict=False)


def hvit_tiny(patch_size=16, **kwargs):
    model = HyperbolicVisionTransformer(
        patch_size=patch_size, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
        qkv_bias=True, **kwargs)
    return model


def hvit_small(patch_size=16, **kwargs):
    model = HyperbolicVisionTransformer(
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, **kwargs)
    return model


def hvit_base(patch_size=16, **kwargs):
    model = HyperbolicVisionTransformer(
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, **kwargs)
    return model
