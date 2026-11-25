# =============================================================================
# dpt.py  —  THE DEPTH ANYTHING V2 MODEL  (most important file for the code review)
# -----------------------------------------------------------------------------
# Architecture = DINOv2 ViT ENCODER  +  DPT DECODER ("DPTHead")
#
#   image (3 x H x W, H and W multiples of 14)
#     -> DINOv2 encoder: split into 14x14 patches -> tokens -> N transformer blocks
#     -> take token outputs from 4 different blocks (shallow ... deep)
#     -> DPT head:
#          (a) REASSEMBLE: tokens -> 2D feature maps at 4 resolutions
#          (b) FUSE: merge them coarse-to-fine with RefineNet-style fusion blocks
#          (c) HEAD: small conv layers -> 1 channel depth map
#   output: 1 x H x W  relative INVERSE depth (larger = closer), non-negative
#
# Running shape example used in the comments below:
#   encoder = 'vitl' (ViT-Large: 24 blocks, 1024-dim tokens), input 518 x 518
#   -> 518 / 14 = 37 patches per side -> 37 * 37 = 1369 patch tokens
# =============================================================================
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose

from .dinov2 import DINOv2                                    # the ViT encoder (Meta's DINOv2)
from .util.blocks import FeatureFusionBlock, _make_scratch    # decoder building blocks
from .util.transform import Resize, NormalizeImage, PrepareForNet  # preprocessing


def _make_fusion_block(features, use_bn, size=None):
    # One RefineNet-style fusion block (see util/blocks.py):
    # adds a skip feature map, refines with residual convs, then upsamples.
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )


class ConvBlock(nn.Module):
    # Conv -> BatchNorm -> ReLU helper. (Defined but NOT used by DepthAnythingV2.)
    def __init__(self, in_feature, out_feature):
        super().__init__()
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_feature, out_feature, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_feature),
            nn.ReLU(True)
        )
    
    def forward(self, x):
        return self.conv_block(x)


# =============================================================================
# DPT DECODER  (Dense Prediction Transformer head, Ranftl et al. 2021)
# Turns 4 sets of ViT tokens into one dense, full-resolution depth map.
# =============================================================================
class DPTHead(nn.Module):
    def __init__(
        self, 
        in_channels,                          # token dimension of the encoder (384 / 768 / 1024 / 1536)
        features=256,                         # common channel width inside the fusion stage
        use_bn=False, 
        out_channels=[256, 512, 1024, 1024],  # channels of the 4 reassembled maps (shallow -> deep)
        use_clstoken=False                    # if True, also mix the [CLS] token into every patch token
    ):
        super(DPTHead, self).__init__()
        
        self.use_clstoken = use_clstoken
        
        # --- REASSEMBLE step 1: 1x1 convs that project the token dim to out_channels[i]
        #     (one per tapped ViT layer). e.g. ViT-L: 1024 -> 256, 512, 1024, 1024
        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])
        
        # --- REASSEMBLE step 2: bring the 4 maps (all 37x37 at first) to 4 DIFFERENT
        #     resolutions, building a feature pyramid like a CNN backbone would have:
        #       layer 1 (shallowest, fine detail):  x4 upsample   37 -> 148
        #       layer 2:                            x2 upsample   37 -> 74
        #       layer 3:                            keep          37
        #       layer 4 (deepest, global context):  x2 downsample 37 -> 19
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(                 # learnable 4x upsampling
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=4,
                stride=4,
                padding=0),
            nn.ConvTranspose2d(                 # learnable 2x upsampling
                in_channels=out_channels[1],
                out_channels=out_channels[1],
                kernel_size=2,
                stride=2,
                padding=0),
            nn.Identity(),                      # unchanged
            nn.Conv2d(                          # stride-2 conv = 2x downsampling
                in_channels=out_channels[3],
                out_channels=out_channels[3],
                kernel_size=3,
                stride=2,
                padding=1)
        ])
        
        # Optional: combine each patch token with the global [CLS] token
        # (concatenate -> Linear -> GELU). Disabled in the released models.
        if use_clstoken:
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(
                        nn.Linear(2 * in_channels, in_channels),
                        nn.GELU()))
        
        # "scratch" holds the fusion part. _make_scratch creates layer1_rn..layer4_rn:
        # 3x3 convs that map each pyramid level to the same width `features` (e.g. 256),
        # so the levels can be added together.
        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )
        
        self.scratch.stem_transpose = None
        
        # --- FUSION: 4 RefineNet blocks, one per pyramid level (refinenet4 = coarsest)
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)
        
        head_features_1 = features
        head_features_2 = 32
        
        # --- OUTPUT HEAD: a few convs that squeeze the features down to 1 channel (depth)
        self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0),  # -> 1 channel = depth
            nn.ReLU(True),        # depth (inverse depth) must be >= 0
            nn.Identity(),
        )
    
    def forward(self, out_features, patch_h, patch_w):
        # out_features: 4 tuples (patch_tokens, cls_token) from 4 ViT blocks
        #   patch_tokens: [B, 1369, 1024]   cls_token: [B, 1024]      (ViT-L, 518x518)
        out = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0]   # released models: just use the patch tokens, ignore [CLS]
            
            # Tokens -> image-like grid: [B, N, C] -> [B, C, N] -> [B, C, 37, 37]
            # (each token goes back to the position of the 14x14 patch it came from)
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            
            x = self.projects[i](x)        # 1x1 conv: change channels   (e.g. 1024 -> 256)
            x = self.resize_layers[i](x)   # change resolution          (e.g. 37 -> 148)
            
            out.append(x)
        
        # Feature pyramid (ViT-L, 518x518):
        #   layer_1 [B, 256, 148, 148]  <- shallow block: fine details, edges
        #   layer_2 [B, 512,  74,  74]
        #   layer_3 [B,1024,  37,  37]
        #   layer_4 [B,1024,  19,  19]  <- deep block: global scene layout / semantics
        layer_1, layer_2, layer_3, layer_4 = out
        
        # Unify channels to `features` (256) with 3x3 convs
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        
        # Coarse-to-fine fusion (like a U-Net decoder): start from the deepest, most global
        # map, upsample it, add the next finer level, refine, repeat.
        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])          # 19 -> 37
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])  # (+layer3) 37 -> 74
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])  # (+layer2) 74 -> 148
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)                             # (+layer1) 148 -> 296 (x2)
        
        out = self.scratch.output_conv1(path_1)                                                     # [B, 128, 296, 296]
        out = F.interpolate(out, (int(patch_h * 14), int(patch_w * 14)), mode="bilinear", align_corners=True)  # -> [B, 128, 518, 518] (network input size)
        out = self.scratch.output_conv2(out)                                                        # -> [B, 1, 518, 518]
        
        return out


# =============================================================================
# FULL MODEL = DINOv2 encoder ("pretrained") + DPT decoder ("depth_head")
# =============================================================================
class DepthAnythingV2(nn.Module):
    def __init__(
        self, 
        encoder='vitl', 
        features=256, 
        out_channels=[256, 512, 1024, 1024], 
        use_bn=False, 
        use_clstoken=False
    ):
        super(DepthAnythingV2, self).__init__()
        
        # Which transformer blocks (0-indexed) feed the decoder: 4 blocks spread evenly
        # through the network so the decoder gets both low-level and high-level features.
        #   ViT-S / ViT-B have 12 blocks -> blocks 3, 6, 9, 12  (1-indexed)
        #   ViT-L has 24 blocks          -> blocks 5, 12, 18, 24
        #   ViT-G has 40 blocks          -> blocks 10, 20, 30, 40
        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11],
            'vitb': [2, 5, 8, 11], 
            'vitl': [4, 11, 17, 23], 
            'vitg': [9, 19, 29, 39]
        }
        
        self.encoder = encoder
        self.pretrained = DINOv2(model_name=encoder)   # ViT encoder (weights come from the checkpoint)
        
        # embed_dim = token size of the encoder (384 / 768 / 1024 / 1536)
        self.depth_head = DPTHead(self.pretrained.embed_dim, features, use_bn, out_channels=out_channels, use_clstoken=use_clstoken)
    
    def forward(self, x):
        # x: normalised image tensor [B, 3, H, W], H and W are multiples of 14
        patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14   # token grid size, e.g. 37 x 37
        
        # Run the ViT and collect (patch_tokens, cls_token) from the 4 chosen blocks
        features = self.pretrained.get_intermediate_layers(x, self.intermediate_layer_idx[self.encoder], return_class_token=True)
        
        depth = self.depth_head(features, patch_h, patch_w)   # [B, 1, H, W]
        depth = F.relu(depth)                                 # make sure it is non-negative
        
        return depth.squeeze(1)                               # [B, H, W] relative inverse depth
    
    @torch.no_grad()   # inference only: no gradients -> faster, less memory
    def infer_image(self, raw_image, input_size=518):
        # raw_image: OpenCV BGR image, uint8, [H, W, 3] of ANY size
        image, (h, w) = self.image2tensor(raw_image, input_size)   # -> [1, 3, H', W'] (H', W' multiples of 14)
        
        depth = self.forward(image)                                 # -> [1, H', W']
        
        # Resize the prediction back to the ORIGINAL image size so it lines up pixel-by-pixel
        depth = F.interpolate(depth[:, None], (h, w), mode="bilinear", align_corners=True)[0, 0]
        
        return depth.cpu().numpy()                                  # [H, W] numpy array
    
    def image2tensor(self, raw_image, input_size=518):        
        # Preprocessing pipeline (same as used in training):
        transform = Compose([
            Resize(
                width=input_size,
                height=input_size,
                resize_target=False,
                keep_aspect_ratio=True,        # do not distort the image
                ensure_multiple_of=14,         # ViT patch size is 14 -> sides must be divisible by 14
                resize_method='lower_bound',   # shorter side becomes input_size (518), longer side scales with it
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            # ImageNet mean/std normalisation (what DINOv2 was pre-trained with)
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),                   # HWC -> CHW, float32
        ])
        
        h, w = raw_image.shape[:2]   # remember the original size
        
        image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0   # BGR -> RGB, scale to 0..1
        
        image = transform({'image': image})['image']
        image = torch.from_numpy(image).unsqueeze(0)                 # add batch dimension -> [1, 3, H', W']
        
        DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        image = image.to(DEVICE)
        
        return image, (h, w)
