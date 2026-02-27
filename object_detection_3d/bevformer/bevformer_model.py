"""
BEVFormer-tiny model in pure PyTorch (no mmcv/mmdet3d dependency).

Architecture:
  - Backbone: ResNet-50 + FPN (output: multi-scale features)
  - BEV Encoder: Transformer encoder with deformable attention
  - Detection Head: DETR-style decoder + classification/regression heads

All operators are standard PyTorch ops, enabling ONNX export.

Reference:
    BEVFormer: Learning Bird's-Eye-View Representation from
    Multi-Camera Images via Spatiotemporal Transformers (ECCV 2022)
    https://arxiv.org/abs/2203.17270
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from deformable_attention import MSDeformAttn


# ============================================================
# Backbone: ResNet-50 + FPN
# ============================================================

class FPN(nn.Module):
    """Feature Pyramid Network with single output level."""

    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_ch in in_channels_list:
            self.lateral_convs.append(
                nn.Conv2d(in_ch, out_channels, 1))
            self.fpn_convs.append(
                nn.Conv2d(out_channels, out_channels, 3, padding=1))

    def forward(self, inputs):
        laterals = [
            conv(x) for x, conv in zip(inputs, self.lateral_convs)
        ]
        # Top-down pathway
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[2:],
                mode='bilinear', align_corners=False)

        outs = [
            conv(lat) for lat, conv in zip(laterals, self.fpn_convs)
        ]
        return outs


class ResNet50Backbone(nn.Module):
    """ResNet-50 backbone extracting multi-scale features."""

    def __init__(self):
        super().__init__()
        resnet = models.resnet50(weights=None)
        self.stem = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1  # stride 4, 256ch
        self.layer2 = resnet.layer2  # stride 8, 512ch
        self.layer3 = resnet.layer3  # stride 16, 1024ch
        self.layer4 = resnet.layer4  # stride 32, 2048ch

    def forward(self, x):
        x = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return [c3, c4, c5]  # stride 8, 16, 32


# ============================================================
# BEV Encoder
# ============================================================

class FFN(nn.Module):
    def __init__(self, d_model, d_ffn, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        x = self.dropout1(F.relu(self.linear1(x)))
        x = self.dropout2(self.linear2(x))
        return self.norm(x + residual)


class BEVEncoderLayer(nn.Module):
    """Single BEV encoder layer with self-attention + cross-attention + FFN."""

    def __init__(self, d_model=256, n_heads=8, d_ffn=512,
                 n_levels=3, n_points=4, dropout=0.1):
        super().__init__()
        # Self-attention on BEV queries
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)

        # Cross-attention: BEV queries attend to image features
        self.cross_attn = MSDeformAttn(
            d_model=d_model, n_levels=n_levels,
            n_heads=n_heads, n_points=n_points)
        self.norm2 = nn.LayerNorm(d_model)

        # Feed-forward
        self.ffn = FFN(d_model, d_ffn, dropout)

    def forward(self, bev_queries, img_feats_flat, ref_points,
                spatial_shapes, bev_pos=None):
        # Self-attention
        q = bev_queries
        if bev_pos is not None:
            q = q + bev_pos
        residual = bev_queries
        bev_queries2, _ = self.self_attn(q, q, bev_queries)
        bev_queries = self.norm1(residual + bev_queries2)

        # Cross-attention to image features
        residual = bev_queries
        q = bev_queries
        if bev_pos is not None:
            q = q + bev_pos
        bev_queries2 = self.cross_attn(
            q, ref_points, img_feats_flat, spatial_shapes)
        bev_queries = self.norm2(residual + bev_queries2)

        # FFN
        bev_queries = self.ffn(bev_queries)

        return bev_queries


class BEVEncoder(nn.Module):
    """BEV Encoder: transforms image features to BEV representation."""

    def __init__(self, d_model=256, n_heads=8, d_ffn=512,
                 n_layers=3, n_levels=3, n_points=4,
                 bev_h=50, bev_w=50, dropout=0.1):
        super().__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.d_model = d_model
        self.n_levels = n_levels

        # Learnable BEV queries
        self.bev_embedding = nn.Embedding(bev_h * bev_w, d_model)

        # Positional embedding for BEV queries
        self.bev_pos = nn.Embedding(bev_h * bev_w, d_model)

        # Encoder layers
        self.layers = nn.ModuleList([
            BEVEncoderLayer(
                d_model, n_heads, d_ffn, n_levels, n_points, dropout)
            for _ in range(n_layers)
        ])

        # Level embedding (added to multi-scale features)
        self.level_embeds = nn.Parameter(
            torch.randn(n_levels, d_model))

    def forward(self, multi_scale_feats):
        """
        Args:
            multi_scale_feats: list of (B, C, H_i, W_i) feature maps

        Returns:
            bev_feat: (B, C, bev_h, bev_w)
        """
        B = multi_scale_feats[0].shape[0]
        device = multi_scale_feats[0].device

        # Flatten multi-scale features
        spatial_shapes = []
        feat_flatten_list = []
        for lvl, feat in enumerate(multi_scale_feats):
            _, C, H, W = feat.shape
            spatial_shapes.append((H, W))
            # (B, C, H, W) -> (B, H*W, C)
            feat_flat = feat.flatten(2).permute(0, 2, 1)
            feat_flat = feat_flat + self.level_embeds[lvl][None, None, :]
            feat_flatten_list.append(feat_flat)

        # (B, sum(Hi*Wi), C)
        img_feats_flat = torch.cat(feat_flatten_list, dim=1)

        # BEV queries
        bev_queries = self.bev_embedding.weight[None].expand(B, -1, -1)
        bev_pos = self.bev_pos.weight[None].expand(B, -1, -1)

        # Reference points for BEV queries: uniform grid in [0, 1]
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, self.bev_h - 0.5, self.bev_h,
                           device=device) / self.bev_h,
            torch.linspace(0.5, self.bev_w - 0.5, self.bev_w,
                           device=device) / self.bev_w,
            indexing='ij'
        )
        ref_points = torch.stack([ref_x.flatten(), ref_y.flatten()], -1)
        # (B, bev_h*bev_w, n_levels, 2)
        ref_points = ref_points[None, :, None, :].expand(
            B, -1, self.n_levels, -1)

        # Encoder layers
        for layer in self.layers:
            bev_queries = layer(
                bev_queries, img_feats_flat, ref_points,
                spatial_shapes, bev_pos)

        # Reshape to (B, C, bev_h, bev_w)
        bev_feat = bev_queries.permute(0, 2, 1).reshape(
            B, self.d_model, self.bev_h, self.bev_w)

        return bev_feat


# ============================================================
# Detection Head
# ============================================================

class DetectionHead(nn.Module):
    """DETR-style detection head for 3D object detection."""

    def __init__(self, d_model=256, num_classes=10,
                 num_queries=300, code_size=10, n_dec_layers=6):
        super().__init__()
        self.num_queries = num_queries
        self.d_model = d_model

        # Object queries
        self.query_embedding = nn.Embedding(num_queries, d_model * 2)

        # Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=8, dim_feedforward=512,
            dropout=0.1, batch_first=True)
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=n_dec_layers)

        # Classification head
        self.cls_branches = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, num_classes),
        )

        # Regression head (cx, cy, cz, w, l, h, sin, cos, vx, vy)
        self.reg_branches = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, code_size),
        )

    def forward(self, bev_feat):
        """
        Args:
            bev_feat: (B, C, bev_h, bev_w)

        Returns:
            cls_scores: (B, num_queries, num_classes)
            bbox_preds: (B, num_queries, code_size)
        """
        B = bev_feat.shape[0]

        # Flatten BEV features as memory for decoder
        memory = bev_feat.flatten(2).permute(0, 2, 1)  # (B, HW, C)

        # Object queries
        query_embed = self.query_embedding.weight
        query_pos = query_embed[:, :self.d_model]
        query = query_embed[:, self.d_model:]
        query_pos = query_pos[None].expand(B, -1, -1)
        query = query[None].expand(B, -1, -1)

        # Decoder
        hs = self.decoder(query + query_pos, memory)  # (B, num_queries, C)

        cls_scores = self.cls_branches(hs)
        bbox_preds = self.reg_branches(hs)

        # Ensure valid bbox ranges with sigmoid/activation
        bbox_preds_center = bbox_preds[..., :3].sigmoid()  # cx, cy, cz in [0,1]
        bbox_preds_rest = bbox_preds[..., 3:]
        bbox_preds = torch.cat([bbox_preds_center, bbox_preds_rest], dim=-1)

        return cls_scores, bbox_preds


# ============================================================
# Full BEVFormer-tiny
# ============================================================

class BEVFormerTiny(nn.Module):
    """BEVFormer-tiny: complete model from images to 3D detections.

    Input:  (B, num_cams, 3, H, W) camera images
    Output: cls_scores (B, num_queries, num_classes),
            bbox_preds (B, num_queries, code_size)
    """

    def __init__(self,
                 num_cams=6,
                 embed_dims=256,
                 num_classes=10,
                 num_queries=300,
                 bev_h=50,
                 bev_w=50,
                 num_enc_layers=3,
                 num_dec_layers=6,
                 fpn_channels=(512, 1024, 2048)):
        super().__init__()
        self.num_cams = num_cams
        self.embed_dims = embed_dims
        self.bev_h = bev_h
        self.bev_w = bev_w

        # Backbone
        self.backbone = ResNet50Backbone()

        # Neck (FPN)
        self.neck = FPN(
            in_channels_list=list(fpn_channels),
            out_channels=embed_dims)

        # BEV Encoder
        self.bev_encoder = BEVEncoder(
            d_model=embed_dims,
            n_heads=8,
            d_ffn=embed_dims * 2,
            n_layers=num_enc_layers,
            n_levels=len(fpn_channels),
            n_points=4,
            bev_h=bev_h,
            bev_w=bev_w)

        # Detection Head
        self.det_head = DetectionHead(
            d_model=embed_dims,
            num_classes=num_classes,
            num_queries=num_queries,
            code_size=10,
            n_dec_layers=num_dec_layers)

    def extract_feat(self, imgs):
        """Extract features from a batch of single-camera images.

        Args:
            imgs: (B*num_cams, 3, H, W)

        Returns:
            list of (B*num_cams, C, H_i, W_i) feature maps
        """
        feats = self.backbone(imgs)
        fpn_feats = self.neck(feats)
        return fpn_feats

    def forward(self, imgs):
        """
        Args:
            imgs: (B, num_cams, 3, H, W)

        Returns:
            cls_scores: (B, num_queries, num_classes)
            bbox_preds: (B, num_queries, code_size)
        """
        B, N, C, H, W = imgs.shape

        # Extract features for all cameras
        imgs_flat = imgs.reshape(B * N, C, H, W)
        multi_scale_feats = self.extract_feat(imgs_flat)

        # Average features across cameras for BEV generation
        # (B*N, C, Hi, Wi) -> (B, N, C, Hi, Wi) -> (B, C, Hi, Wi)
        averaged_feats = []
        for feat in multi_scale_feats:
            _, Cf, Hf, Wf = feat.shape
            feat_per_batch = feat.reshape(B, N, Cf, Hf, Wf)
            feat_avg = feat_per_batch.mean(dim=1)
            averaged_feats.append(feat_avg)

        # BEV encoding
        bev_feat = self.bev_encoder(averaged_feats)

        # Detection
        cls_scores, bbox_preds = self.det_head(bev_feat)

        return cls_scores, bbox_preds


def build_bevformer_tiny(num_cams=6, img_h=480, img_w=800):
    """Build a BEVFormer-tiny model with default config."""
    model = BEVFormerTiny(
        num_cams=num_cams,
        embed_dims=256,
        num_classes=10,
        num_queries=300,
        bev_h=50,
        bev_w=50,
        num_enc_layers=3,
        num_dec_layers=6,
    )
    return model
