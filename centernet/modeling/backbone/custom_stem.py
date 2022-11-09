import numpy as np
import fvcore.nn.weight_init as weight_init
import torch
import torch.nn.functional as F
from torch import nn

from detectron2.modeling import BACKBONE_REGISTRY, ShapeSpec
from detectron2.layers import (
    CNNBlockBase,
    Conv2d,
    DeformConv,
    ModulatedDeformConv,
    ShapeSpec,
    get_norm,
)

class CustomStem(CNNBlockBase):
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