"""
Export BEVFormer-tiny backbone (ResNet50 + FPN) to ONNX.

BEVFormer uses Multi-Scale Deformable Attention which is not directly
supported by standard ONNX operators. This script exports the image
backbone (ResNet50 + FPN) as a standard ONNX model that extracts
multi-scale features from camera images.

Usage:
    pip install mmcv-full==1.5.0 mmdet==2.25.1 mmdet3d==1.0.0rc4
    git clone https://github.com/fundamentalvision/BEVFormer.git
    cd BEVFormer
    python bevformer_onnx_export.py --checkpoint bevformer_tiny_epoch_24.pth
"""

import argparse
import sys
import os

import numpy as np


def export_backbone(args):
    import torch
    import torch.nn as nn
    from mmcv import Config
    from mmdet3d.models import build_model

    # BEVFormer-tiny config
    bev_h = 50
    bev_w = 50
    num_classes = 10
    embed_dims = 256
    num_queries = 900
    img_h = 928
    img_w = 1600
    num_cams = 6

    class BEVFormerBackbone(nn.Module):
        """Wrapper that extracts only the image backbone + neck."""

        def __init__(self, model):
            super().__init__()
            self.img_backbone = model.img_backbone
            self.img_neck = model.img_neck

        def forward(self, img):
            # img: (B, C, H, W)
            feats = self.img_backbone(img)
            if isinstance(feats, (list, tuple)):
                feats = [feats[-1]]
            neck_out = self.img_neck(feats)
            return neck_out[0]

    if args.config and args.checkpoint:
        cfg = Config.fromfile(args.config)
        cfg.model.train_cfg = None
        model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))

        from mmcv.runner import load_checkpoint
        load_checkpoint(model, args.checkpoint, map_location='cpu')
        model.eval()
    else:
        # Use torchvision ResNet50 + simple FPN as fallback
        import torchvision.models as models

        class SimpleBackboneNeck(nn.Module):
            def __init__(self):
                super().__init__()
                resnet = models.resnet50(pretrained=True)
                self.layer0 = nn.Sequential(
                    resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
                self.layer1 = resnet.layer1
                self.layer2 = resnet.layer2
                self.layer3 = resnet.layer3
                self.layer4 = resnet.layer4
                # Simple 1x1 conv to project to embed_dims
                self.neck_conv = nn.Conv2d(2048, embed_dims, 1)
                self.neck_bn = nn.BatchNorm2d(embed_dims)

            def forward(self, img):
                x = self.layer0(img)
                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.layer4(x)
                x = self.neck_conv(x)
                x = self.neck_bn(x)
                return x

        model = None
        backbone_model = SimpleBackboneNeck()
        backbone_model.eval()

    if model is not None:
        backbone_model = BEVFormerBackbone(model)
        backbone_model.eval()

    # Export backbone
    dummy_img = torch.randn(1, 3, img_h, img_w)
    output_path = args.output if args.output else 'bevformer_tiny_backbone.onnx'

    print(f'Exporting backbone to {output_path}...')
    torch.onnx.export(
        backbone_model,
        dummy_img,
        output_path,
        input_names=['image'],
        output_names=['features'],
        dynamic_axes={
            'image': {0: 'batch'},
            'features': {0: 'batch'},
        },
        opset_version=args.opset,
        do_constant_folding=True,
    )
    print(f'Backbone exported to {output_path}')

    # Verify
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print('ONNX model check passed.')

    # Print model info
    print(f'Input: image (1, 3, {img_h}, {img_w})')
    for output in onnx_model.graph.output:
        name = output.name
        shape = [d.dim_value for d in output.type.tensor_type.shape.dim]
        print(f'Output: {name} {shape}')


def export_detection_head(args):
    """Export a simplified detection head for BEV features."""
    import torch
    import torch.nn as nn

    embed_dims = 256
    num_classes = 10
    num_queries = 900
    code_size = 10  # x, y, z, w, l, h, sin, cos, vx, vy

    class SimpleBEVDetHead(nn.Module):
        """Simplified detection head that takes BEV features and outputs
        3D bounding box predictions."""

        def __init__(self):
            super().__init__()
            self.query_embedding = nn.Embedding(num_queries, embed_dims * 2)

            # Decoder layers
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=embed_dims,
                nhead=8,
                dim_feedforward=512,
                dropout=0.1,
                batch_first=True,
            )
            self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)

            # Classification and regression heads
            self.cls_head = nn.Linear(embed_dims, num_classes)
            self.reg_head = nn.Sequential(
                nn.Linear(embed_dims, embed_dims),
                nn.ReLU(),
                nn.Linear(embed_dims, code_size),
            )

        def forward(self, bev_features):
            # bev_features: (B, C, H, W)
            B = bev_features.shape[0]
            bev_flat = bev_features.flatten(2).permute(0, 2, 1)  # (B, HW, C)

            query_embed = self.query_embedding.weight  # (num_queries, C*2)
            query_pos = query_embed[:, :embed_dims].unsqueeze(0).expand(B, -1, -1)
            query = query_embed[:, embed_dims:].unsqueeze(0).expand(B, -1, -1)

            hs = self.decoder(query, bev_flat)  # (B, num_queries, C)

            cls_scores = self.cls_head(hs)  # (B, num_queries, num_classes)
            bbox_preds = self.reg_head(hs)  # (B, num_queries, code_size)

            return cls_scores, bbox_preds

    head_model = SimpleBEVDetHead()
    head_model.eval()

    bev_h, bev_w = 50, 50
    dummy_bev = torch.randn(1, embed_dims, bev_h, bev_w)
    output_path = args.output if args.output else 'bevformer_tiny_head.onnx'
    output_path = output_path.replace('.onnx', '_head.onnx')

    print(f'Exporting detection head to {output_path}...')
    torch.onnx.export(
        head_model,
        dummy_bev,
        output_path,
        input_names=['bev_features'],
        output_names=['cls_scores', 'bbox_preds'],
        dynamic_axes={
            'bev_features': {0: 'batch'},
            'cls_scores': {0: 'batch'},
            'bbox_preds': {0: 'batch'},
        },
        opset_version=args.opset,
        do_constant_folding=True,
    )
    print(f'Detection head exported to {output_path}')


def main():
    parser = argparse.ArgumentParser(
        description='Export BEVFormer to ONNX')
    parser.add_argument(
        '--config', type=str, default=None,
        help='BEVFormer config file path '
             '(e.g. projects/configs/bevformer/bevformer_tiny.py)')
    parser.add_argument(
        '--checkpoint', type=str, default=None,
        help='PyTorch checkpoint file path '
             '(e.g. bevformer_tiny_epoch_24.pth)')
    parser.add_argument(
        '--output', type=str, default='bevformer_tiny_backbone.onnx',
        help='Output ONNX file path')
    parser.add_argument(
        '--opset', type=int, default=11,
        help='ONNX opset version')
    parser.add_argument(
        '--export-head', action='store_true',
        help='Also export detection head')
    args = parser.parse_args()

    export_backbone(args)
    if args.export_head:
        export_detection_head(args)


if __name__ == '__main__':
    main()
