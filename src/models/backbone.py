"""
SpikeConformer - Conformer Backbone Architecture
=================================================
Dual-branch architecture: CNN (local features) + Transformer (global context)
with cross-attention fusion at multiple scales.

Paper: Energy-Efficient Ear Biometric Recognition via Conformer-Guided SNNs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Tuple

from .layers import (
    DepthwiseSeparableConvBlock,
    TransformerEncoderLayer,
    CrossAttentionFusion,
    PatchEmbedding,
)


class Conformer(nn.Module):
    """
    Conformer backbone for ear biometric recognition.

    Architecture:
        - Patch embedding (16x16, d=384)
        - Parallel CNN branch (DWSepConv, 3 stages) and Transformer branch (6 layers)
        - Cross-attention fusion at 3 interaction points
        - Global average pool + classification head
    """

    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (3, 224, 224),
        num_classes: int = 164,
        embed_dim: int = 384,
        cnn_channels: list = None,
        transformer_layers: int = 6,
        num_heads: int = 6,
        ffn_expansion: int = 4,
        num_fusion_points: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        if cnn_channels is None:
            cnn_channels = [96, 192, 384]

        self.input_shape = input_shape
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        in_channels = input_shape[0]

        # Patch embedding for Transformer branch
        self.patch_embed = PatchEmbedding(
            in_channels=in_channels,
            embed_dim=embed_dim,
            patch_size=16,
            image_size=input_shape[1],
        )
        num_patches = self.patch_embed.num_patches  # 196 for 224x224

        # CNN Branch: 3 stages of depthwise separable conv blocks
        self.cnn_stages = nn.ModuleList()
        cnn_in = in_channels
        for ch in cnn_channels:
            self.cnn_stages.append(
                DepthwiseSeparableConvBlock(cnn_in, ch, stride=2 if cnn_in != in_channels else 2)
            )
            cnn_in = ch

        # Transformer Branch: 6 layers grouped into 3 stages of 2
        layers_per_stage = transformer_layers // num_fusion_points
        self.transformer_stages = nn.ModuleList()
        for _ in range(num_fusion_points):
            stage = nn.ModuleList([
                TransformerEncoderLayer(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    ffn_expansion=ffn_expansion,
                    dropout=dropout,
                )
                for _ in range(layers_per_stage)
            ])
            self.transformer_stages.append(stage)

        # Cross-attention fusion modules
        self.fusion_modules = nn.ModuleList([
            CrossAttentionFusion(cnn_dim=ch, trans_dim=embed_dim)
            for ch in cnn_channels
        ])

        # CNN projection to match embed_dim for fusion
        self.cnn_projections = nn.ModuleList([
            nn.Conv2d(ch, embed_dim, 1) for ch in cnn_channels
        ])

        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head_drop = nn.Dropout(dropout)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, 3, 224, 224)

        Returns:
            Logits of shape (B, num_classes)
        """
        B = x.shape[0]

        # Initialize both branches
        # Transformer branch: patch embedding
        # (B, 3, 224, 224) -> (B, num_patches, embed_dim)
        trans_tokens = self.patch_embed(x)

        # CNN branch starts from raw input
        cnn_feat = x  # (B, 3, 224, 224)

        # Process through 3 stages with fusion
        for i, (cnn_stage, trans_stage, fusion, cnn_proj) in enumerate(zip(
            self.cnn_stages, self.transformer_stages, self.fusion_modules, self.cnn_projections
        )):
            # CNN stage: (B, C_in, H, W) -> (B, C_out, H/2, W/2)
            cnn_feat = cnn_stage(cnn_feat)

            # Transformer stage: (B, N, D) -> (B, N, D)
            for layer in trans_stage:
                trans_tokens = layer(trans_tokens)

            # Project CNN features to embed_dim for fusion
            # (B, C, H', W') -> (B, embed_dim, H', W')
            cnn_proj_feat = cnn_proj(cnn_feat)

            # Cross-attention fusion
            # CNN queries attend to Transformer keys/values
            trans_tokens = fusion(cnn_proj_feat, trans_tokens)

        # Classification: global average pool over token dimension
        # (B, N, D) -> (B, D)
        cls_token = trans_tokens.mean(dim=1)
        cls_token = self.norm(cls_token)
        cls_token = self.head_drop(cls_token)

        # (B, D) -> (B, num_classes)
        logits = self.head(cls_token)
        return logits

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Extract embeddings before classification head (for ArcFace)."""
        B = x.shape[0]
        trans_tokens = self.patch_embed(x)
        cnn_feat = x

        for i, (cnn_stage, trans_stage, fusion, cnn_proj) in enumerate(zip(
            self.cnn_stages, self.transformer_stages, self.fusion_modules, self.cnn_projections
        )):
            cnn_feat = cnn_stage(cnn_feat)
            for layer in trans_stage:
                trans_tokens = layer(trans_tokens)
            cnn_proj_feat = cnn_proj(cnn_feat)
            trans_tokens = fusion(cnn_proj_feat, trans_tokens)

        cls_token = trans_tokens.mean(dim=1)
        cls_token = self.norm(cls_token)
        return cls_token

    def get_param_count(self) -> dict:
        """Return parameter count per module."""
        counts = {}
        counts["patch_embed"] = sum(p.numel() for p in self.patch_embed.parameters())
        counts["cnn_stages"] = sum(p.numel() for p in self.cnn_stages.parameters())
        counts["transformer_stages"] = sum(p.numel() for p in self.transformer_stages.parameters())
        counts["fusion_modules"] = sum(p.numel() for p in self.fusion_modules.parameters())
        counts["cnn_projections"] = sum(p.numel() for p in self.cnn_projections.parameters())
        counts["head"] = sum(p.numel() for p in self.head.parameters())
        counts["total"] = sum(p.numel() for p in self.parameters())
        counts["trainable"] = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return counts


def build_model(cfg: dict) -> nn.Module:
    """Factory function to build Conformer from config dict."""
    return Conformer(
        input_shape=tuple(cfg["input_shape"]),
        num_classes=cfg["num_classes"],
        embed_dim=cfg["embed_dim"],
        cnn_channels=cfg.get("cnn_channels", [96, 192, 384]),
        transformer_layers=cfg.get("transformer_layers", 6),
        num_heads=cfg.get("num_heads", 6),
        ffn_expansion=cfg.get("ffn_expansion", 4),
        num_fusion_points=cfg.get("num_fusion_points", 3),
        dropout=cfg.get("dropout", 0.3),
    )
