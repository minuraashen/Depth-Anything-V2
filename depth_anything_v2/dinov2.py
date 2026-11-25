# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/main/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

# =============================================================================
# dinov2.py  —  THE ENCODER: DINOv2 Vision Transformer (ViT), code from Meta
# -----------------------------------------------------------------------------
# Why DINOv2? It was pre-trained with self-supervision on 142M curated images, so its
# features generalise very well. The paper (Sec. 3, App. B.6) found that only DINOv2
# (and only the Giant size) transfers well from synthetic to real images, which is
# why the teacher is DINOv2-G and every student uses a DINOv2 encoder.
#
# How a ViT processes an image (ViT-L, 518 x 518 input):
#   1. PatchEmbed: cut into 14x14 patches -> 37 x 37 = 1369 patches -> each becomes a
#      1024-dim vector ("token")                              -> [B, 1369, 1024]
#   2. prepend a learnable [CLS] token                       -> [B, 1370, 1024]
#   3. add positional embeddings (so tokens know where they are in the image)
#   4. 24 transformer blocks (self-attention + MLP), shape stays [B, 1370, 1024]
# For depth we do NOT use the final classification output. Instead
# get_intermediate_layers() returns the tokens after 4 chosen blocks.
# =============================================================================

from functools import partial
import math
import logging
from typing import Sequence, Tuple, Union, Callable

import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch.nn.init import trunc_normal_

from .dinov2_layers import Mlp, PatchEmbed, SwiGLUFFNFused, MemEffAttention, NestedTensorBlock as Block


logger = logging.getLogger("dinov2")


def named_apply(fn: Callable, module: nn.Module, name="", depth_first=True, include_root=False) -> nn.Module:
    # Helper: recursively apply fn to every sub-module (used for weight initialisation)
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(fn=fn, module=child_module, name=child_name, depth_first=depth_first, include_root=True)
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


class BlockChunk(nn.ModuleList):
    # Groups several blocks together (only used for distributed training with FSDP;
    # Depth Anything sets block_chunks=0, so this is not used)
    def forward(self, x):
        for b in self:
            x = b(x)
        return x


class DinoVisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        ffn_bias=True,
        proj_bias=True,
        drop_path_rate=0.0,
        drop_path_uniform=False,
        init_values=None,  # for layerscale: None or 0 => no layerscale
        embed_layer=PatchEmbed,
        act_layer=nn.GELU,
        block_fn=Block,
        ffn_layer="mlp",
        block_chunks=1,
        num_register_tokens=0,
        interpolate_antialias=False,
        interpolate_offset=0.1,
    ):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            proj_bias (bool): enable bias for proj in attn if True
            ffn_bias (bool): enable bias for ffn if True
            drop_path_rate (float): stochastic depth rate
            drop_path_uniform (bool): apply uniform drop rate across blocks
            weight_init (str): weight init scheme
            init_values (float): layer-scale init values
            embed_layer (nn.Module): patch embedding layer
            act_layer (nn.Module): MLP activation layer
            block_fn (nn.Module): transformer block class
            ffn_layer (str): "mlp", "swiglu", "swiglufused" or "identity"
            block_chunks: (int) split block sequence into block_chunks units for FSDP wrap
            num_register_tokens: (int) number of extra cls tokens (so-called "registers")
            interpolate_antialias: (str) flag to apply anti-aliasing when interpolating positional embeddings
            interpolate_offset: (float) work-around offset to apply when interpolating positional embeddings
        """
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 1          # just the [CLS] token
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size # 14 for Depth Anything
        self.num_register_tokens = num_register_tokens
        self.interpolate_antialias = interpolate_antialias
        self.interpolate_offset = interpolate_offset

        # 1) Patch embedding: image -> sequence of patch tokens
        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches   # 37*37 = 1369 for a 518x518 image

        # 2) Learnable [CLS] token and learnable positional embeddings (one per patch + one for CLS)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        assert num_register_tokens >= 0
        # "Register" tokens (DINOv2-reg variant). Depth Anything V2 uses 0 registers: the paper
        # (App. B.6) found the register version worse for ViT-G.
        self.register_tokens = (
            nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim)) if num_register_tokens else None
        )

        # Stochastic depth (randomly skipping blocks during training); rate is 0 here
        if drop_path_uniform is True:
            dpr = [drop_path_rate] * depth
        else:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        # Choose the feed-forward network inside each block:
        # standard MLP for S/B/L, SwiGLU variant for Giant
        if ffn_layer == "mlp":
            logger.info("using MLP layer as FFN")
            ffn_layer = Mlp
        elif ffn_layer == "swiglufused" or ffn_layer == "swiglu":
            logger.info("using SwiGLU layer as FFN")
            ffn_layer = SwiGLUFFNFused
        elif ffn_layer == "identity":
            logger.info("using Identity layer as FFN")

            def f(*args, **kwargs):
                return nn.Identity()

            ffn_layer = f
        else:
            raise NotImplementedError

        # 3) The stack of `depth` transformer blocks (12 for S/B, 24 for L, 40 for G)
        blocks_list = [
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                ffn_layer=ffn_layer,
                init_values=init_values,
            )
            for i in range(depth)
        ]
        if block_chunks > 0:
            self.chunked_blocks = True
            chunked_blocks = []
            chunksize = depth // block_chunks
            for i in range(0, depth, chunksize):
                # this is to keep the block index consistent if we chunk the block list
                chunked_blocks.append([nn.Identity()] * i + blocks_list[i : i + chunksize])
            self.blocks = nn.ModuleList([BlockChunk(p) for p in chunked_blocks])
        else:
            # Depth Anything path (block_chunks=0): a plain list of blocks
            self.chunked_blocks = False
            self.blocks = nn.ModuleList(blocks_list)

        self.norm = norm_layer(embed_dim)   # final LayerNorm
        self.head = nn.Identity()           # no classification head

        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))   # used only in DINOv2's masked pre-training

        self.init_weights()

    def init_weights(self):
        # Random initialisation; irrelevant at inference because checkpoint weights overwrite it
        trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-6)
        if self.register_tokens is not None:
            nn.init.normal_(self.register_tokens, std=1e-6)
        named_apply(init_weights_vit_timm, self)

    def interpolate_pos_encoding(self, x, w, h):
        # The positional embeddings were learned for a 37x37 grid (518x518 input).
        # For any other input size (e.g. 924x518 -> 66x37 grid) the grid of position
        # vectors is resized with BICUBIC interpolation, so the model accepts any
        # resolution / aspect ratio (as long as sides are multiples of 14).
        # (Note: here `w` = image dim 2 and `h` = image dim 3, see prepare_tokens_with_masks.)
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed   # same grid as training -> use as is
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]     # position vector of the [CLS] token (not resized)
        patch_pos_embed = pos_embed[:, 1:]    # position vectors of the 37*37 patches
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        # DINOv2 with register modify the interpolate_offset from 0.1 to 0.0
        w0, h0 = w0 + self.interpolate_offset, h0 + self.interpolate_offset
        # w0, h0 = w0 + 0.1, h0 + 0.1
        
        sqrt_N = math.sqrt(N)
        sx, sy = float(w0) / sqrt_N, float(h0) / sqrt_N
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(sqrt_N), int(sqrt_N), dim).permute(0, 3, 1, 2),   # as a 2D grid [1, dim, 37, 37]
            scale_factor=(sx, sy),
            # (int(w0), int(h0)), # to solve the upsampling shape issue
            mode="bicubic",
            antialias=self.interpolate_antialias
        )
        
        assert int(w0) == patch_pos_embed.shape[-2]
        assert int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)   # back to a token sequence
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)

    def prepare_tokens_with_masks(self, x, masks=None):
        # Image -> token sequence ready for the transformer blocks
        B, nc, w, h = x.shape
        x = self.patch_embed(x)                    # [B, 3, 518, 518] -> [B, 1369, 1024]
        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)

        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)   # prepend [CLS] -> [B, 1370, 1024]
        x = x + self.interpolate_pos_encoding(x, w, h)                          # add position information

        if self.register_tokens is not None:
            x = torch.cat(
                (
                    x[:, :1],
                    self.register_tokens.expand(x.shape[0], -1, -1),
                    x[:, 1:],
                ),
                dim=1,
            )

        return x

    # forward_features / forward_features_list / forward: the standard DINOv2 interface
    # (final-layer outputs). Depth Anything does not call these; it calls
    # get_intermediate_layers() below.
    def forward_features_list(self, x_list, masks_list):
        x = [self.prepare_tokens_with_masks(x, masks) for x, masks in zip(x_list, masks_list)]
        for blk in self.blocks:
            x = blk(x)

        all_x = x
        output = []
        for x, masks in zip(all_x, masks_list):
            x_norm = self.norm(x)
            output.append(
                {
                    "x_norm_clstoken": x_norm[:, 0],
                    "x_norm_regtokens": x_norm[:, 1 : self.num_register_tokens + 1],
                    "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1 :],
                    "x_prenorm": x,
                    "masks": masks,
                }
            )
        return output

    def forward_features(self, x, masks=None):
        if isinstance(x, list):
            return self.forward_features_list(x, masks)

        x = self.prepare_tokens_with_masks(x, masks)

        for blk in self.blocks:
            x = blk(x)

        x_norm = self.norm(x)
        return {
            "x_norm_clstoken": x_norm[:, 0],
            "x_norm_regtokens": x_norm[:, 1 : self.num_register_tokens + 1],
            "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1 :],
            "x_prenorm": x,
            "masks": masks,
        }

    def _get_intermediate_layers_not_chunked(self, x, n=1):
        # Run ALL blocks in order, but SAVE the output of the requested ones.
        # e.g. ViT-L with n=[4, 11, 17, 23] -> outputs of blocks 5, 12, 18, 24 (1-indexed)
        x = self.prepare_tokens_with_masks(x)
        # If n is an int, take the n last blocks. If it's a list, take them
        output, total_block_len = [], len(self.blocks)
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in blocks_to_take:
                output.append(x)
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def _get_intermediate_layers_chunked(self, x, n=1):
        # Same as above for the chunked (FSDP) layout; not used by Depth Anything
        x = self.prepare_tokens_with_masks(x)
        output, i, total_block_len = [], 0, len(self.blocks[-1])
        # If n is an int, take the n last blocks. If it's a list, take them
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for block_chunk in self.blocks:
            for blk in block_chunk[i:]:  # Passing the nn.Identity()
                x = blk(x)
                if i in blocks_to_take:
                    output.append(x)
                i += 1
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: Union[int, Sequence] = 1,  # Layers or n last layers to take
        reshape: bool = False,
        return_class_token: bool = False,
        norm=True
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
        # >>> This is the method Depth Anything calls (from dpt.py) <<<
        # Returns, for each of the 4 chosen blocks, a tuple (patch_tokens, cls_token):
        #   patch_tokens [B, 1369, 1024], cls_token [B, 1024]   (ViT-L, 518x518)
        if self.chunked_blocks:
            outputs = self._get_intermediate_layers_chunked(x, n)
        else:
            outputs = self._get_intermediate_layers_not_chunked(x, n)
        if norm:
            outputs = [self.norm(out) for out in outputs]          # apply the final LayerNorm to each
        class_tokens = [out[:, 0] for out in outputs]               # token 0 = [CLS]
        outputs = [out[:, 1 + self.num_register_tokens:] for out in outputs]   # the rest = patch tokens
        if reshape:
            B, _, w, h = x.shape
            outputs = [
                out.reshape(B, w // self.patch_size, h // self.patch_size, -1).permute(0, 3, 1, 2).contiguous()
                for out in outputs
            ]
        if return_class_token:
            return tuple(zip(outputs, class_tokens))
        return tuple(outputs)

    def forward(self, *args, is_training=False, **kwargs):
        ret = self.forward_features(*args, **kwargs)
        if is_training:
            return ret
        else:
            return self.head(ret["x_norm_clstoken"])


def init_weights_vit_timm(module: nn.Module, name: str = ""):
    """ViT weight initialization, original timm impl (for reproducibility)"""
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


# -----------------------------------------------------------------------------
# The 4 encoder sizes used by Depth Anything V2
#   size   | token dim | blocks | heads | (dim per head = 64 in all cases)
#   Small  |    384    |   12   |   6   |
#   Base   |    768    |   12   |  12   |
#   Large  |   1024    |   24   |  16   |
#   Giant  |   1536    |   40   |  24   |  <- the TEACHER in the paper
# -----------------------------------------------------------------------------
def vit_small(patch_size=16, num_register_tokens=0, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vit_base(patch_size=16, num_register_tokens=0, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vit_large(patch_size=16, num_register_tokens=0, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vit_giant2(patch_size=16, num_register_tokens=0, **kwargs):
    """
    Close to ViT-giant, with embed-dim 1536 and 24 heads => embed-dim per head 64
    """
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1536,
        depth=40,
        num_heads=24,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def DINOv2(model_name):
    # Factory called from dpt.py: DINOv2('vitl') etc.
    model_zoo = {
        "vits": vit_small, 
        "vitb": vit_base, 
        "vitl": vit_large, 
        "vitg": vit_giant2
    }
    
    return model_zoo[model_name](
        img_size=518,               # default/training resolution -> 37x37 positional grid
        patch_size=14,              # 14x14 pixel patches (this is why inputs must be multiples of 14)
        init_values=1.0,            # LayerScale enabled
        ffn_layer="mlp" if model_name != "vitg" else "swiglufused",
        block_chunks=0,
        num_register_tokens=0,      # original DINOv2 (no registers)
        interpolate_antialias=False,
        interpolate_offset=0.1
    )
