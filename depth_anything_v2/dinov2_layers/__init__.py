# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Building blocks of the DINOv2 Vision Transformer (used by ../dinov2.py):
#   PatchEmbed      - image -> patch tokens
#   MemEffAttention - multi-head self-attention
#   Mlp / SwiGLU    - feed-forward network inside each block
#   NestedTensorBlock - one transformer block (attention + MLP, both with residuals)
from .mlp import Mlp
from .patch_embed import PatchEmbed
from .swiglu_ffn import SwiGLUFFN, SwiGLUFFNFused
from .block import NestedTensorBlock
from .attention import MemEffAttention
