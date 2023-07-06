# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
This file provides the definition of the convolutional heads used to predict masks, as well as the losses
"""
from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:
    from panopticapi.utils import id2rgb, rgb2id
except ImportError:
    pass


def _expand(tensor, length: int):
    return tensor.unsqueeze(1).repeat(1, int(length), 1, 1, 1).flatten(0, 1)


class UnetBlock(nn.Module):
    def __init__(self, up_in, x_in, n_out):
        super().__init__()
        up_out = x_out = n_out // 2
        self.x_conv = nn.Conv2d(x_in, x_out, 1)
        self.tr_conv = nn.ConvTranspose2d(up_in, up_out, 2, stride=2)

        self.bn = nn.BatchNorm2d(n_out)

    def forward(self, up_p, x_p):
        up_p = self.tr_conv(up_p)
        x_p = self.x_conv(x_p)
        cat_p = torch.cat([up_p, x_p], dim=1)
        return self.bn(F.relu(cat_p))


class MaskHeadSmallConv_UNet(nn.Module):
    def __init__(self, dim, fpn_dims, context_dim, num_classes):
        super().__init__()
        self.num_classes = num_classes

        self.up1 = UnetBlock(dim, 256, 256)
        self.up2 = UnetBlock(256, 128, 256)
        self.up3 = UnetBlock(256, 64, 256)
        self.up4 = UnetBlock(256, 64, 256)

        self.up5 = nn.ConvTranspose2d(256, self.num_classes, 2, stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor, bbox_mask: Tensor, fpns: List[Tensor]):
        x = torch.cat([_expand(x, bbox_mask.shape[1]), bbox_mask.flatten(0, 1)], 1)
        x = F.relu(x)

        skip_feature = fpns[0]
        if skip_feature.size(0) != x.size(0):
            skip_feature = _expand(skip_feature, x.size(0) // skip_feature.size(0))
        x = self.up1(x, skip_feature)

        skip_feature = fpns[1]
        if skip_feature.size(0) != x.size(0):
            skip_feature = _expand(skip_feature, x.size(0) // skip_feature.size(0))
        x = self.up2(x, skip_feature)

        skip_feature = fpns[2]
        if skip_feature.size(0) != x.size(0):
            skip_feature = _expand(skip_feature, x.size(0) // skip_feature.size(0))
        x = self.up3(x, skip_feature)

        skip_feature = fpns[3]
        if skip_feature.size(0) != x.size(0):
            skip_feature = _expand(skip_feature, x.size(0) // skip_feature.size(0))
        x = self.up4(x, skip_feature)

        x = self.up5(x)
        return x


class MHAttentionMap(nn.Module):
    """This is a 2D attention module, which only returns the attention softmax (no multiplication by value)"""

    def __init__(self, query_dim, hidden_dim, num_heads, dropout=0.0, bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

        self.q_linear = nn.Linear(query_dim, hidden_dim, bias=bias)
        self.k_linear = nn.Linear(query_dim, hidden_dim, bias=bias)

        nn.init.zeros_(self.k_linear.bias)
        nn.init.zeros_(self.q_linear.bias)
        nn.init.xavier_uniform_(self.k_linear.weight)
        nn.init.xavier_uniform_(self.q_linear.weight)
        self.normalize_fact = float(hidden_dim / self.num_heads) ** -0.5

    def forward(self, q, k, mask: Optional[Tensor] = None):
        q = self.q_linear(q)
        k = F.conv2d(k, self.k_linear.weight.unsqueeze(-1).unsqueeze(-1), self.k_linear.bias)
        qh = q.view(q.shape[0], q.shape[1], self.num_heads, self.hidden_dim // self.num_heads)
        kh = k.view(k.shape[0], self.num_heads, self.hidden_dim // self.num_heads, k.shape[-2], k.shape[-1])
        weights = torch.einsum("bqnc,bnchw->bqnhw", qh * self.normalize_fact, kh)

        if mask is not None:
            weights.masked_fill_(mask.unsqueeze(1).unsqueeze(1), float("-inf"))
        weights = F.softmax(weights.flatten(2), dim=-1).view(weights.size())
        weights = self.dropout(weights)
        return weights
