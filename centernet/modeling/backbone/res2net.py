# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# This file is modified from https://github.com/Res2Net/Res2Net-detectron2/blob/master/detectron2/modeling/backbone/resnet.py
# The original file is under Apache-2.0 License
import numpy as np
import fvcore.nn.weight_init as weight_init
import torch
import torch.nn.functional as F
from torch import nn
import pickle

from detectron2.layers import (
    CNNBlockBase,
    Conv2d,
    DeformConv,
    ModulatedDeformConv,
    ShapeSpec,
    get_norm,
)

from detectron2.modeling.backbone import Backbone
from detectron2.modeling.backbone.fpn import FPN 
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from .fpn_p5 import LastLevelP6P7_P5
from .bifpn import BiFPN


__all__ = [
    "ResNetBlockBase",
    "BasicBlock",
    "BottleneckBlock",
    "DeformBottleneckBlock",
    "BasicStem",
    "ResNet",
    "make_stage",
    "build_res2net_backbone",
]


ResNetBlockBase = CNNBlockBase
"""
Alias for backward compatibiltiy.
"""


class BasicBlock(CNNBlockBase):
    """
    The basic residual block for ResNet-18 and ResNet-34, with two 3x3 conv layers
    and a projection shortcut if needed.
    """

    def __init__(self, in_channels, out_channels, *, stride=1, norm="BN"):
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            stride (int): Stride for the first conv.
            norm (str or callable): normalization for all conv layers.
                See :func:`layers.get_norm` for supported format.
        """
        super().__init__(in_channels, out_channels, stride)

        if in_channels != out_channels:
            self.shortcut = Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
                norm=get_norm(norm, out_channels),
            )
        else:
            self.shortcut = None

        self.conv1 = Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
            norm=get_norm(norm, out_channels),
        )

        self.conv2 = Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            norm=get_norm(norm, out_channels),
        )

        for layer in [self.conv1, self.conv2, self.shortcut]:
            if layer is not None:  # shortcut can be None
                weight_init.c2_msra_fill(layer)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu_(out)
        out = self.conv2(out)

        if self.shortcut is not None:
            shortcut = self.shortcut(x)
        else:
            shortcut = x

        out += shortcut
        out = F.relu_(out)
        return out


class BottleneckBlock(CNNBlockBase):
    """
    The standard bottle2neck residual block used by Res2Net-50, 101 and 152.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        *,
        bottleneck_channels,
        stride=1,
        num_groups=1,
        norm="BN",
        stride_in_1x1=False,
        dilation=1,
        basewidth=26, 
        scale=4,
    ):
        """
        Args:
            bottleneck_channels (int): number of output channels for the 3x3
                "bottleneck" conv layers.
            num_groups (int): number of groups for the 3x3 conv layer.
            norm (str or callable): normalization for all conv layers.
                See :func:`layers.get_norm` for supported format.
            stride_in_1x1 (bool): when stride>1, whether to put stride in the
                first 1x1 convolution or the bottleneck 3x3 convolution.
            dilation (int): the dilation rate of the 3x3 conv layer.
        """
        super().__init__(in_channels, out_channels, stride)

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.AvgPool2d(kernel_size=stride, stride=stride, 
                    ceil_mode=True, count_include_pad=False),
                Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=1,
                    bias=False,
                    norm=get_norm(norm, out_channels),
                )
            )
        else:
            self.shortcut = None

        # The original MSRA ResNet models have stride in the first 1x1 conv
        # The subsequent fb.torch.resnet and Caffe2 ResNe[X]t implementations have
        # stride in the 3x3 conv
        stride_1x1, stride_3x3 = (stride, 1) if stride_in_1x1 else (1, stride)
        width = bottleneck_channels//scale

        self.conv1 = Conv2d(
            in_channels,
            bottleneck_channels,
            kernel_size=1,
            stride=stride_1x1,
            bias=False,
            norm=get_norm(norm, bottleneck_channels),
        )
        if scale == 1:
          self.nums = 1
        else:
          self.nums = scale -1
        if self.in_channels!=self.out_channels and stride_3x3!=2:
            self.pool = nn.AvgPool2d(kernel_size=3, stride = stride_3x3, padding=1)

        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(nn.Conv2d(
                            width, 
                            width, 
                            kernel_size=3, 
                            stride=stride_3x3, 
                            padding=1 * dilation, 
                            bias=False,
                            groups=num_groups,
                            dilation=dilation,
                            ))
            bns.append(get_norm(norm, width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = Conv2d(
            bottleneck_channels,
            out_channels,
            kernel_size=1,
            bias=False,
            norm=get_norm(norm, out_channels),
        )
        self.scale = scale
        self.width = width
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride_3x3 = stride_3x3
        for layer in [self.conv1, self.conv3]:
            if layer is not None:  # shortcut can be None
                weight_init.c2_msra_fill(layer)
        if self.shortcut is not None:
            for layer in self.shortcut.modules():
                if isinstance(layer, Conv2d):
                    weight_init.c2_msra_fill(layer)
                
        for layer in self.convs:
            if layer is not None:  # shortcut can be None
                weight_init.c2_msra_fill(layer)

        # Zero-initialize the last normalization in each residual branch,
        # so that at the beginning, the residual branch starts with zeros,
        # and each residual block behaves like an identity.
        # See Sec 5.1 in "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour":
        # "For BN layers, the learnable scaling coefficient γ is initialized
        # to be 1, except for each residual block's last BN
        # where γ is initialized to be 0."

        # nn.init.constant_(self.conv3.norm.weight, 0)
        # TODO this somehow hurts performance when training GN models from scratch.
        # Add it as an option when we need to use this code to train a backbone.

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu_(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i==0 or self.in_channels!=self.out_channels:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = F.relu_(self.bns[i](sp))
            if i==0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        if self.scale!=1 and self.stride_3x3==1:
            out = torch.cat((out, spx[self.nums]), 1)
        elif self.scale != 1 and self.stride_3x3==2:
            out = torch.cat((out, self.pool(spx[self.nums])), 1)

        out = self.conv3(out)

        if self.shortcut is not None:
            shortcut = self.shortcut(x)
        else:
            shortcut = x

        out += shortcut
        out = F.relu_(out)
        return out


class DeformBottleneckBlock(ResNetBlockBase):
    """
    Not implemented for res2net yet.
    Similar to :class:`BottleneckBlock`, but with deformable conv in the 3x3 convolution.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        *,
        bottleneck_channels,
        stride=1,
        num_groups=1,
        norm="BN",
        stride_in_1x1=False,
        dilation=1,
        deform_modulated=False,
        deform_num_groups=1,
        basewidth=26, 
        scale=4,
    ):
        super().__init__(in_channels, out_channels, stride)
        self.deform_modulated = deform_modulated

        if in_channels != out_channels:
            # self.shortcut = Conv2d(
            #     in_channels,
            #     out_channels,
            #     kernel_size=1,
            #     stride=stride,
            #     bias=False,
            #     norm=get_norm(norm, out_channels),
            # )
            self.shortcut = nn.Sequential(
                nn.AvgPool2d(kernel_size=stride, stride=stride, 
                    ceil_mode=True, count_include_pad=False),
                Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=1,
                    bias=False,
                    norm=get_norm(norm, out_channels),
                )
            )
        else:
            self.shortcut = None

        stride_1x1, stride_3x3 = (stride, 1) if stride_in_1x1 else (1, stride)
        width = bottleneck_channels//scale

        self.conv1 = Conv2d(
            in_channels,
            bottleneck_channels,
            kernel_size=1,
            stride=stride_1x1,
            bias=False,
            norm=get_norm(norm, bottleneck_channels),
        )

        if scale == 1:
          self.nums = 1
        else:
          self.nums = scale -1
        if self.in_channels!=self.out_channels and stride_3x3!=2:
            self.pool = nn.AvgPool2d(kernel_size=3, stride = stride_3x3, padding=1)

        if deform_modulated:
            deform_conv_op = ModulatedDeformConv
            # offset channels are 2 or 3 (if with modulated) * kernel_size * kernel_size
            offset_channels = 27
        else:
            deform_conv_op = DeformConv
            offset_channels = 18

        # self.conv2_offset = Conv2d(
        #     bottleneck_channels,
        #     offset_channels * deform_num_groups,
        #     kernel_size=3,
        #     stride=stride_3x3,
        #     padding=1 * dilation,
        #     dilation=dilation,
        # )
        # self.conv2 = deform_conv_op(
        #     bottleneck_channels,
        #     bottleneck_channels,
        #     kernel_size=3,
        #     stride=stride_3x3,
        #     padding=1 * dilation,
        #     bias=False,
        #     groups=num_groups,
        #     dilation=dilation,
        #     deformable_groups=deform_num_groups,
        #     norm=get_norm(norm, bottleneck_channels),
        # )

        conv2_offsets = []
        convs = []
        bns = []
        for i in range(self.nums):
            conv2_offsets.append(Conv2d(
                            width, 
                            offset_channels * deform_num_groups, 
                            kernel_size=3, 
                            stride=stride_3x3, 
                            padding=1 * dilation, 
                            bias=False,
                            groups=num_groups,
                            dilation=dilation,
                            ))
            convs.append(deform_conv_op(
                            width, 
                            width, 
                            kernel_size=3, 
                            stride=stride_3x3, 
                            padding=1 * dilation, 
                            bias=False,
                            groups=num_groups,
                            dilation=dilation,
                            deformable_groups=deform_num_groups,
                            ))
            bns.append(get_norm(norm, width))
        self.conv2_offsets = nn.ModuleList(conv2_offsets)
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = Conv2d(
            bottleneck_channels,
            out_channels,
            kernel_size=1,
            bias=False,
            norm=get_norm(norm, out_channels),
        )
        self.scale = scale
        self.width = width
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride_3x3 = stride_3x3
        # for layer in [self.conv1, self.conv2, self.conv3, self.shortcut]:
        #     if layer is not None:  # shortcut can be None
        #         weight_init.c2_msra_fill(layer)

        # nn.init.constant_(self.conv2_offset.weight, 0)
        # nn.init.constant_(self.conv2_offset.bias, 0)
        for layer in [self.conv1, self.conv3]:
            if layer is not None:  # shortcut can be None
                weight_init.c2_msra_fill(layer)
        if self.shortcut is not None:
            for layer in self.shortcut.modules():
                if isinstance(layer, Conv2d):
                    weight_init.c2_msra_fill(layer)
                
        for layer in self.convs:
            if layer is not None:  # shortcut can be None
                weight_init.c2_msra_fill(layer)

        for layer in self.conv2_offsets:
            if layer.weight is not None:
                nn.init.constant_(layer.weight, 0)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu_(out)

        # if self.deform_modulated:
        #     offset_mask = self.conv2_offset(out)
        #     offset_x, offset_y, mask = torch.chunk(offset_mask, 3, dim=1)
        #     offset = torch.cat((offset_x, offset_y), dim=1)
        #     mask = mask.sigmoid()
        #     out = self.conv2(out, offset, mask)
        # else:
        #     offset = self.conv2_offset(out)
        #     out = self.conv2(out, offset)
        # out = F.relu_(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i==0 or self.in_channels!=self.out_channels:
                sp = spx[i].contiguous()
            else:
                sp = sp + spx[i].contiguous()
            
            # sp = self.convs[i](sp)
            if self.deform_modulated:
                offset_mask = self.conv2_offsets[i](sp)
                offset_x, offset_y, mask = torch.chunk(offset_mask, 3, dim=1)
                offset = torch.cat((offset_x, offset_y), dim=1)
                mask = mask.sigmoid()
                sp = self.convs[i](sp, offset, mask)
            else:
                offset = self.conv2_offsets[i](sp)
                sp = self.convs[i](sp, offset)
            sp = F.relu_(self.bns[i](sp))
            if i==0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        if self.scale!=1 and self.stride_3x3==1:
            out = torch.cat((out, spx[self.nums]), 1)
        elif self.scale != 1 and self.stride_3x3==2:
            out = torch.cat((out, self.pool(spx[self.nums])), 1)

        out = self.conv3(out)

        if self.shortcut is not None:
            shortcut = self.shortcut(x)
        else:
            shortcut = x

        out += shortcut
        out = F.relu_(out)
        return out


def make_stage(block_class, num_blocks, first_stride, *, in_channels, out_channels, **kwargs):
    """
    Create a list of blocks just like those in a ResNet stage.
    Args:
        block_class (type): a subclass of ResNetBlockBase
        num_blocks (int):
        first_stride (int): the stride of the first block. The other blocks will have stride=1.
        in_channels (int): input channels of the entire stage.
        out_channels (int): output channels of **every block** in the stage.
        kwargs: other arguments passed to the constructor of every block.
    Returns:
        list[nn.Module]: a list of block module.
    """
    assert "stride" not in kwargs, "Stride of blocks in make_stage cannot be changed."
    blocks = []
    for i in range(num_blocks):
        blocks.append(
            block_class(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=first_stride if i == 0 else 1,
                **kwargs,
            )
        )
        in_channels = out_channels
    return blocks


class BasicStem(CNNBlockBase):
    """
    The standard ResNet stem (layers before the first residual block).
    """

    def __init__(self, in_channels=3, out_channels=64, norm="BN"):
        """
        Args:
            norm (str or callable): norm after the first conv layer.
                See :func:`layers.get_norm` for supported format.
        """
        super().__init__(in_channels, out_channels, 4)
        self.in_channels = in_channels
        self.conv1 = nn.Sequential(
            Conv2d(
                in_channels,
                32,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
                ),
            get_norm(norm, 32),
            nn.ReLU(inplace=True),
            Conv2d(
                32,
                32,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                ),
            get_norm(norm, 32),
            nn.ReLU(inplace=True),
            Conv2d(
                32,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                ),
        )
        self.bn1 = get_norm(norm, out_channels)

        for layer in self.conv1:
            if isinstance(layer, Conv2d):
                weight_init.c2_msra_fill(layer)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu_(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        return x


class customConv2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.register_buffer('identity_kernel', torch.ones(out_channels, in_channels, *kernel_size))
        self.weights = nn.Parameter(torch.Tensor(out_channels, in_channels, *kernel_size), requires_grad=True)
        with torch.no_grad():
            self.weights.data.normal_(0.0, 0.8)

    def forward(self, img):
        b, c, h, w = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
        p00 = 0.0
        p01 = -0.000287
        p10 = 0.0
        p11 = 0.266
        p20 = 0.0
        p21 = -0.1097
        p30 = 0.0
        img_unf = nn.functional.unfold(img, kernel_size=self.kernel_size,
                                       stride=self.stride, padding=self.padding).transpose(1, 2).contiguous()
        self.identity_kernel = self.identity_kernel.contiguous()
        identity_weights = self.identity_kernel.view(self.identity_kernel.size(0), -1).contiguous()
        self.weights = self.weights.contiguous()
        weights = self.weights.view(self.weights.size(0), -1).contiguous()

        # f0 = (p00 + torch.zeros_like(img_unf)).matmul(identity_weights.t())
        # f1 = (p10 * (img_unf - 0.5)).matmul(identity_weights.t())
        # f2 = (p01 * torch.ones_like(img_unf)).matmul(weights.t())
        # f3 = (p20 * torch.pow(img_unf - 0.5, 2)).matmul(identity_weights.t())
        # f4 = (p11 * (img_unf - 0.5)).matmul(weights.t())
        # f5 = (p30 * torch.pow(img_unf - 0.5, 3)).matmul(identity_weights.t())
        # f6 = (p21 * torch.pow(img_unf - 0.5, 2)).matmul(weights.t())
        # f = (f0 + f1 + f2 + f3 + f4 + f5 + f6).transpose(1, 2)

        f = ((p00 + torch.zeros_like(img_unf) +
              p10 * (img_unf) +
              p20 * torch.pow(img_unf, 2) +
              p30 * torch.pow(img_unf, 3)).matmul(identity_weights.t()) +
             (p01 * torch.ones_like(img_unf) +
              p11 * (img_unf) +
              p21 * torch.pow(img_unf, 2)
              ).matmul(weights.t().contiguous())).transpose(1, 2).contiguous() / 75

        g1 = ((0.11 / 75 + torch.zeros_like(img_unf)).matmul(identity_weights.t()) +
              (0.001309 * torch.ones_like(img_unf) +
               0.00619 * (img_unf) - 0.009 * torch.pow(img_unf, 2) + 0.001383 * torch.pow(img_unf, 3)
               ).matmul(weights.t().contiguous())).transpose(1, 2).contiguous()

        g2 = ((0.179 / 75 + torch.zeros_like(img_unf)).matmul(identity_weights.t()) +
              (-0.0025 * torch.ones_like(img_unf) +
               0.00303 * (img_unf) - 0.00484 * torch.pow(img_unf, 2) + 0.0175 * torch.pow(img_unf, 3)
               ).matmul(weights.t().contiguous())).transpose(1, 2).contiguous()

        g3 = ((0.238 / 75 + torch.zeros_like(img_unf)).matmul(identity_weights.t()) +
              (-0.000954 * torch.ones_like(img_unf) +
               0.00187 * (img_unf) + 0.001877 * torch.pow(img_unf, 2) + 0.01502 * torch.pow(img_unf, 3)
               ).matmul(weights.t().contiguous())).transpose(1, 2).contiguous()

        g4 = ((0.388 / 75 + torch.zeros_like(img_unf)).matmul(identity_weights.t()) +
              (-0.00734 * torch.ones_like(img_unf) +
               0.001117 * (img_unf) + 0.00752 * torch.pow(img_unf, 2) + 0.009 * torch.pow(img_unf, 3)
               ).matmul(weights.t().contiguous())).transpose(1, 2).contiguous()

        g5 = ((0.507 / 75 + torch.zeros_like(img_unf)).matmul(identity_weights.t()) +
              (-0.01017 * torch.ones_like(img_unf) +
               0.000426 * (img_unf) + 0.00837 * torch.pow(img_unf, 2) + 0.00413 * torch.pow(img_unf, 3)
               ).matmul(weights.t().contiguous())).transpose(1, 2).contiguous()

        s1 = 1 / (1 + torch.exp(-10 * (0.15 - f)))
        s2 = 1 / (1 + torch.exp(-10 * (f - 0.15))) + 1 / (1 + torch.exp(-10 * (0.23 - f))) - 1
        s3 = 1 / (1 + torch.exp(-10 * (f - 0.23))) + 1 / (1 + torch.exp(-10 * (0.32 - f))) - 1
        s4 = 1 / (1 + torch.exp(-10 * (f - 0.32))) + 1 / (1 + torch.exp(-10 * (0.39 - f))) - 1
        s5 = 1 / (1 + torch.exp(-10 * (f - 0.39)))

        out = s1 * g1 + s2 * g2 + s3 * g3 + s4 * g4 + s5 * g5

        out_xshape = int((h - self.kernel_size[0] + 2 * self.padding) / self.stride) + 1
        out_yshape = int((w - self.kernel_size[1] + 2 * self.padding) / self.stride) + 1
        # out = f.contiguous()
        out = out.view(b, self.out_channels, out_xshape, out_yshape)  # .contiguous()
        # out = out/(3*self.kernel_size[0]*self.kernel_size[1])
        return out


class CustomStem(BasicStem):
    """
    The custom ResNet stem (layers before the first residual block).
    """
    def __init__(self, in_channels=3, out_channels=64, norm="BN", noise_level=0):
        """
        Args:
            norm (str or callable): norm after the first conv layer.
                See :func:`layers.get_norm` for supported format.
        """
        super().__init__(in_channels, out_channels, norm)
        self.custom_conv = customConv2(in_channels, out_channels, kernel_size=(3,3), stride=1, padding=0)
        self.noise_level = noise_level

    def quantize(self, x, k, do_quantise=True):
        xmax = torch.max(x)
        xmin = torch.min(x)
        if xmax < -xmin:
            xmax = -xmin
        if not do_quantise:
            return x
        digital = torch.round(((2 ** k) - 1) * x / xmax)
        output = xmax * digital / ((2 ** k) - 1)
        return output

    def noise(self, x, mean=0.0, std=0.01):
        f = torch.normal(torch.full(x.size(), mean), std).to(x.device)
        x += x * torch.max(x) * f
        return x

    def save_pickle(self, x, path):
        try:
            with open(path, "rb") as f:
                pkl = pickle.load(f)
            pkl = torch.cat((pkl,x.cpu()), dim=0)
        except Exception as e:
            print(e)
            pkl = x.cpu()
        finally:
            print("saving", path)
            with open(path, "wb") as f:
                pickle.dump(pkl, f)

    def forward(self, x):
        #self.save_pickle(x, path="output/pre_stem_cpu.pkl")
        x = self.conv1(x)  # Change this to custom_conv
        #x = self.custom_conv(x)
        x = self.bn1(x)
        x = F.relu_(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        # Add noise and quantize
        x = self.noise(x, std=self.noise_level)
        x = self.quantize(x, k=8)
        #self.save_pickle(x, path="output/post_stem_cpu.pkl")
        return x


class ResNet(Backbone):
    def __init__(self, stem, stages, num_classes=None, out_features=None):
        """
        Args:
            stem (nn.Module): a stem module
            stages (list[list[CNNBlockBase]]): several (typically 4) stages,
                each contains multiple :class:`CNNBlockBase`.
            num_classes (None or int): if None, will not perform classification.
                Otherwise, will create a linear layer.
            out_features (list[str]): name of the layers whose outputs should
                be returned in forward. Can be anything in "stem", "linear", or "res2" ...
                If None, will return the output of the last layer.
        """
        super(ResNet, self).__init__()
        self.stem = stem
        self.num_classes = num_classes

        current_stride = self.stem.stride
        self._out_feature_strides = {"stem": current_stride}
        self._out_feature_channels = {"stem": self.stem.out_channels}

        self.stages_and_names = []
        for i, blocks in enumerate(stages):
            assert len(blocks) > 0, len(blocks)
            for block in blocks:
                assert isinstance(block, CNNBlockBase), block

            name = "res" + str(i + 2)
            stage = nn.Sequential(*blocks)

            self.add_module(name, stage)
            self.stages_and_names.append((stage, name))

            self._out_feature_strides[name] = current_stride = int(
                current_stride * np.prod([k.stride for k in blocks])
            )
            self._out_feature_channels[name] = curr_channels = blocks[-1].out_channels

        if num_classes is not None:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.linear = nn.Linear(curr_channels, num_classes)

            # Sec 5.1 in "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour":
            # "The 1000-way fully-connected layer is initialized by
            # drawing weights from a zero-mean Gaussian with standard deviation of 0.01."
            nn.init.normal_(self.linear.weight, std=0.01)
            name = "linear"

        if out_features is None:
            out_features = [name]
        self._out_features = out_features
        assert len(self._out_features)
        children = [x[0] for x in self.named_children()]
        for out_feature in self._out_features:
            assert out_feature in children, "Available children: {}".format(", ".join(children))

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        if "stem" in self._out_features:
            outputs["stem"] = x
        for stage, name in self.stages_and_names:
            x = stage(x)
            if name in self._out_features:
                outputs[name] = x
        if self.num_classes is not None:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.linear(x)
            if "linear" in self._out_features:
                outputs["linear"] = x
        return outputs

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }

    def freeze(self, freeze_at=0):
        """
        Freeze the first several stages of the ResNet. Commonly used in
        fine-tuning.
        Args:
            freeze_at (int): number of stem and stages to freeze.
                `1` means freezing the stem. `2` means freezing the stem and
                the first stage, etc.
        Returns:
            nn.Module: this ResNet itself
        """
        if freeze_at >= 1:
            self.stem.freeze()
        for idx, (stage, _) in enumerate(self.stages_and_names, start=2):
            if freeze_at >= idx:
                for block in stage.children():
                    block.freeze()
        return self


@BACKBONE_REGISTRY.register()
def build_res2net_backbone(cfg, input_shape):
    """
    Create a Res2Net instance from config.
    Returns:
        ResNet: a :class:`ResNet` instance.
    """
    # need registration of new blocks/stems?
    norm = cfg.MODEL.RESNETS.NORM
    stem = CustomStem(
        in_channels=input_shape.channels,
        out_channels=cfg.MODEL.RESNETS.STEM_OUT_CHANNELS,
        norm=norm,
        noise_level=cfg.MODEL.RESNETS.NOISE_LEVEL
    )

    # fmt: off
    freeze_at           = cfg.MODEL.BACKBONE.FREEZE_AT
    out_features        = cfg.MODEL.RESNETS.OUT_FEATURES
    depth               = cfg.MODEL.RESNETS.DEPTH
    num_groups          = cfg.MODEL.RESNETS.NUM_GROUPS
    width_per_group     = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
    scale              = 4
    bottleneck_channels = num_groups * width_per_group * scale
    in_channels         = cfg.MODEL.RESNETS.STEM_OUT_CHANNELS
    out_channels        = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    stride_in_1x1       = cfg.MODEL.RESNETS.STRIDE_IN_1X1
    res5_dilation       = cfg.MODEL.RESNETS.RES5_DILATION
    deform_on_per_stage = cfg.MODEL.RESNETS.DEFORM_ON_PER_STAGE
    deform_modulated    = cfg.MODEL.RESNETS.DEFORM_MODULATED
    deform_num_groups   = cfg.MODEL.RESNETS.DEFORM_NUM_GROUPS
    # fmt: on
    assert res5_dilation in {1, 2}, "res5_dilation cannot be {}.".format(res5_dilation)

    num_blocks_per_stage = {
        18: [2, 2, 2, 2],
        34: [3, 4, 6, 3],
        50: [3, 4, 6, 3],
        101: [3, 4, 23, 3],
        152: [3, 8, 36, 3],
    }[depth]

    if depth in [18, 34]:
        assert out_channels == 64, "Must set MODEL.RESNETS.RES2_OUT_CHANNELS = 64 for R18/R34"
        assert not any(
            deform_on_per_stage
        ), "MODEL.RESNETS.DEFORM_ON_PER_STAGE unsupported for R18/R34"
        assert res5_dilation == 1, "Must set MODEL.RESNETS.RES5_DILATION = 1 for R18/R34"
        assert num_groups == 1, "Must set MODEL.RESNETS.NUM_GROUPS = 1 for R18/R34"

    stages = []

    # Avoid creating variables without gradients
    # It consumes extra memory and may cause allreduce to fail
    out_stage_idx = [{"res2": 2, "res3": 3, "res4": 4, "res5": 5}[f] for f in out_features]
    max_stage_idx = max(out_stage_idx)
    for idx, stage_idx in enumerate(range(2, max_stage_idx + 1)):
        dilation = res5_dilation if stage_idx == 5 else 1
        first_stride = 1 if idx == 0 or (stage_idx == 5 and dilation == 2) else 2
        stage_kargs = {
            "num_blocks": num_blocks_per_stage[idx],
            "first_stride": first_stride,
            "in_channels": in_channels,
            "out_channels": out_channels,
            "norm": norm,
        }
        # Use BasicBlock for R18 and R34.
        if depth in [18, 34]:
            stage_kargs["block_class"] = BasicBlock
        else:
            stage_kargs["bottleneck_channels"] = bottleneck_channels
            stage_kargs["stride_in_1x1"] = stride_in_1x1
            stage_kargs["dilation"] = dilation
            stage_kargs["num_groups"] = num_groups
            stage_kargs["scale"] = scale

            if deform_on_per_stage[idx]:
                stage_kargs["block_class"] = DeformBottleneckBlock
                stage_kargs["deform_modulated"] = deform_modulated
                stage_kargs["deform_num_groups"] = deform_num_groups
            else:
                stage_kargs["block_class"] = BottleneckBlock
        blocks = make_stage(**stage_kargs)
        in_channels = out_channels
        out_channels *= 2
        bottleneck_channels *= 2
        stages.append(blocks)
    return ResNet(stem, stages, out_features=out_features).freeze(freeze_at)


@BACKBONE_REGISTRY.register()
def build_p67_res2net_fpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode

    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = build_res2net_backbone(cfg, input_shape)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelP6P7_P5(out_channels, out_channels),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    return backbone


@BACKBONE_REGISTRY.register()
def build_res2net_bifpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode

    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = build_res2net_backbone(cfg, input_shape)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    backbone = BiFPN(
        cfg=cfg,
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=cfg.MODEL.BIFPN.OUT_CHANNELS,
        norm=cfg.MODEL.BIFPN.NORM,
        num_levels=cfg.MODEL.BIFPN.NUM_LEVELS,
        num_bifpn=cfg.MODEL.BIFPN.NUM_BIFPN,
        separable_conv=cfg.MODEL.BIFPN.SEPARABLE_CONV,
    )
    return backbone