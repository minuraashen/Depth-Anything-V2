# =============================================================================
# blocks.py  —  building blocks of the DPT decoder's FUSION stage
# -----------------------------------------------------------------------------
#   _make_scratch      : 3x3 convs that give every pyramid level the same channel width
#   ResidualConvUnit   : two 3x3 convs with a skip connection (refines features)
#   FeatureFusionBlock : "RefineNet" block = add skip level + refine + upsample
# Used by DPTHead in dpt.py.
# =============================================================================
import torch.nn as nn


def _make_scratch(in_shape, out_shape, groups=1, expand=False):
    # in_shape : channels of the 4 reassembled maps, e.g. [256, 512, 1024, 1024]
    # out_shape: common width `features`, e.g. 256
    # Returns a container module with layer1_rn ... layer4_rn ("rn" = RefineNet input).
    scratch = nn.Module()

    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    if len(in_shape) >= 4:
        out_shape4 = out_shape

    # expand=True would give deeper levels more channels; Depth Anything uses expand=False
    # (all levels -> the same width), so this branch is not used.
    if expand:
        out_shape1 = out_shape
        out_shape2 = out_shape * 2
        out_shape3 = out_shape * 4
        if len(in_shape) >= 4:
            out_shape4 = out_shape * 8

    # One 3x3 conv per level: in_shape[i] channels -> out_shape channels (no resolution change)
    scratch.layer1_rn = nn.Conv2d(in_shape[0], out_shape1, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    scratch.layer2_rn = nn.Conv2d(in_shape[1], out_shape2, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    scratch.layer3_rn = nn.Conv2d(in_shape[2], out_shape3, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    if len(in_shape) >= 4:
        scratch.layer4_rn = nn.Conv2d(in_shape[3], out_shape4, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)

    return scratch


class ResidualConvUnit(nn.Module):
    """Residual convolution module.
    """
    # out = x + conv(ReLU(conv(ReLU(x))))   -- same shape in and out.
    # The skip connection lets the block learn only a correction to its input.

    def __init__(self, features, activation, bn):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()

        self.bn = bn   # BatchNorm off in Depth Anything (use_bn=False)

        self.groups=1

        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)
        
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)

        if self.bn == True:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)

        self.activation = activation   # ReLU

        # FloatFunctional.add is just "a + b", written this way so the model can be quantized
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """
        
        out = self.activation(x)   # pre-activation style: ReLU before each conv
        out = self.conv1(out)
        if self.bn == True:
            out = self.bn1(out)
       
        out = self.activation(out)
        out = self.conv2(out)
        if self.bn == True:
            out = self.bn2(out)

        if self.groups > 1:
            out = self.conv_merge(out)   # never reached (groups is always 1)

        return self.skip_add.add(out, x)   # residual connection: out + x


class FeatureFusionBlock(nn.Module):
    """Feature fusion block.
    """
    # One step of the coarse-to-fine decoder:
    #   output = Upsample( RCU2( coarse_path + RCU1(skip_feature) ) ) -> 1x1 conv
    # xs[0] = the coarser path coming from the previous fusion block (already at this resolution)
    # xs[1] = the skip feature map of this pyramid level (missing for the very first/coarsest block)

    def __init__(
        self, 
        features, 
        activation, 
        deconv=False, 
        bn=False, 
        expand=False, 
        align_corners=True,
        size=None
    ):
        """Init.
        
        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock, self).__init__()

        self.deconv = deconv
        self.align_corners = align_corners

        self.groups=1

        self.expand = expand
        out_features = features
        if self.expand == True:
            out_features = features // 2
        
        self.out_conv = nn.Conv2d(features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=1)

        self.resConfUnit1 = ResidualConvUnit(features, activation, bn)   # refines the skip feature
        self.resConfUnit2 = ResidualConvUnit(features, activation, bn)   # refines the merged result
        
        self.skip_add = nn.quantized.FloatFunctional()

        self.size=size

    def forward(self, *xs, size=None):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]   # coarser path (from deeper layers)

        if len(xs) == 2:
            # merge: coarse path + refined skip feature from this level
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)

        output = self.resConfUnit2(output)

        # Upsample: to an explicit `size` (the next finer level's resolution) if given,
        # otherwise by a factor of 2 (used by the last block, refinenet1)
        if (size is None) and (self.size is None):
            modifier = {"scale_factor": 2}
        elif size is None:
            modifier = {"size": self.size}
        else:
            modifier = {"size": size}

        output = nn.functional.interpolate(output, **modifier, mode="bilinear", align_corners=self.align_corners)
        
        output = self.out_conv(output)   # 1x1 conv (channels unchanged since expand=False)

        return output
