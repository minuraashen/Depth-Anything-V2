# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

# Multi-head SELF-ATTENTION: every token looks at every other token, so each patch
# can use information from the whole image (global context -> good scene layout).
#   Attention(Q, K, V) = softmax(Q K^T / sqrt(d_head)) V      computed per head
# MemEffAttention = same maths, but uses xFormers' memory-efficient kernel when it is
# installed; otherwise it falls back to the plain Attention.forward below.

import logging

from torch import Tensor
from torch import nn


logger = logging.getLogger("dinov2")


try:
    from xformers.ops import memory_efficient_attention, unbind, fmha

    XFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("xFormers not available")
    XFORMERS_AVAILABLE = False


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads          # e.g. 1024 / 16 heads = 64
        self.scale = head_dim**-0.5          # 1 / sqrt(d_head)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)   # one Linear produces Q, K and V together
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape   # batch, number of tokens (1370), token dim (1024)
        # -> [3, B, heads, N, d_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1)      # [B, heads, N, N]: similarity of every token pair

        attn = attn.softmax(dim=-1)         # attention weights (each row sums to 1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)   # weighted sum of values, heads concatenated
        x = self.proj(x)                                  # output projection
        x = self.proj_drop(x)
        return x


class MemEffAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None) -> Tensor:
        if not XFORMERS_AVAILABLE:
            # no xFormers installed -> use the standard attention above (same result)
            assert attn_bias is None, "xFormers is required for nested tensors usage"
            return super().forward(x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

        