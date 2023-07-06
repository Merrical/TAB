# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import LowRankMultivariateNormal
from utils.distributions import ReshapedDistribution

from utils.misc import NestedTensor, nested_tensor_from_tensor_list
from .backbone import build_backbone
from .segmentation import MHAttentionMap, MaskHeadSmallConv_UNet
from .transformer import build_transformer


class TAB(nn.Module):
    def __init__(self, backbone, transformer, num_queries, aux_loss=False, num_classes=1, rank=5):
        super().__init__()
        self.num_queries = num_queries
        self.rater_num = self.num_queries - 1
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.num_classes = num_classes
        self.rank = rank

        nheads = self.transformer.nhead
        self.bbox_attention = MHAttentionMap(hidden_dim, hidden_dim, nheads, dropout=0.0)
        self.mask_head = MaskHeadSmallConv_UNet(hidden_dim + nheads, [256, 128, 64, 64], hidden_dim, 32)

        self.bnout = nn.BatchNorm2d(32)
        self.mu_head = nn.Conv2d(32, self.num_classes, 1)
        self.diagonal_head = nn.Conv2d(32, self.num_classes, 1)
        self.factor_head = nn.Conv2d(32, self.num_classes * self.rank, 1)

    def forward(self, samples: NestedTensor, training=True):
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)
        bs = features[-1].tensors.shape[0]

        src, mask = features[-1].decompose()
        assert mask is not None
        src_proj = self.input_proj(src)
        hs, memory = self.transformer(src_proj, mask, self.query_embed.weight, pos[-1])

        bbox_mask = self.bbox_attention(hs[-1], memory, mask=mask)
        seg_masks = self.mask_head(src_proj, bbox_mask, [features[3].tensors, features[2].tensors, features[1].tensors, features[0].tensors])
        seg_feature = F.relu(self.bnout(seg_masks))

        event_shape = (self.num_classes,) + seg_feature.data.cpu().numpy().shape[2:]
        mus = self.mu_head(seg_feature)
        diagonals = torch.abs(self.diagonal_head(seg_feature)) + 1e-5
        factors = self.factor_head(seg_feature)

        mus = mus.view(bs, self.num_queries, self.num_classes * mus.shape[-2] * mus.shape[-1]).to(torch.float64)
        diagonals = diagonals.view(bs, self.num_queries, self.num_classes * diagonals.shape[-2] * diagonals.shape[-1]).to(torch.float64)
        factors = factors.view(bs, self.num_queries, self.num_classes * factors.shape[-2] * factors.shape[-1], self.rank).to(torch.float64)

        global_dist = LowRankMultivariateNormal(loc=mus[:, 0], cov_factor=factors[:, 0], cov_diag=diagonals[:, 0], validate_args=False)
        global_dist = ReshapedDistribution(global_dist, event_shape)
        rater_dists = list()
        for i in range(self.rater_num):
            dist_temp = LowRankMultivariateNormal(loc=mus[:, i+1], cov_factor=factors[:, i+1], cov_diag=diagonals[:, i+1], validate_args=False)
            rater_dists.append(ReshapedDistribution(dist_temp, event_shape))

        global_mu, global_var = global_dist.mean, global_dist.variance
        rater_mus, rater_vars = list(), list()
        for i in range(self.rater_num):
            rater_mus.append(rater_dists[i].mean)
            rater_vars.append(rater_dists[i].variance)
        rater_mus = torch.stack(rater_mus, dim=1)
        rater_vars = torch.stack(rater_vars, dim=1)

        if training:
            rater_samples = [dist.rsample() for dist in rater_dists]
            global_samples = [global_dist.rsample() for i in range(self.rater_num)]
        else:
            rater_samples = [dist.sample() for dist in rater_dists]
            global_samples = [global_dist.sample() for i in range(self.rater_num)]
        rater_samples = torch.stack(rater_samples, dim=1)
        global_samples = torch.stack(global_samples, dim=1)

        return global_mu, rater_mus, global_var, rater_vars, rater_samples, global_samples


def build_TAB_model(args):
    backbone = build_backbone(args)
    transformer = build_transformer(args)
    model = TAB(
        backbone,
        transformer,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
        num_classes=args.num_classes,
        rank=args.rank
    )
    return model
