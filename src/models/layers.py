"""
Custom layers for SpikeConformer.
- DepthwiseSeparableConvBlock (CNN branch)
- TransformerEncoderLayer (Transformer branch)
- CrossAttentionFusion (feature interaction)
- PatchEmbedding (input tokenization)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math
from typing import Optional


class PatchEmbedding(nn.Module):
    """Convert image to patch tokens with positional encoding."""

    def __init__(self, in_channels: int = 3, embed_dim: int = 384,
                 patch_size: int = 16, image_size: int = 224):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2  # 196 for 224/16
        self.embed_dim = embed_dim

        # Linear projection of flattened patches
        self.projection = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )
        # Learnable positional encoding
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches, embed_dim) * 0.02
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            tokens: (B, num_patches, embed_dim)
        """
        # (B, C, H, W) -> (B, embed_dim, H/P, W/P)
        x = self.projection(x)
        # (B, embed_dim, H/P, W/P) -> (B, num_patches, embed_dim)
        x = rearrange(x, "b d h w -> b (h w) d")
        # Add positional encoding
        x = x + self.pos_embed
        return x


class DepthwiseSeparableConvBlock(nn.Module):
    """
    Depthwise separable convolution block with residual connection.
    DWConv(3x3) -> BN -> GELU -> PWConv(1x1) -> BN -> GELU + residual
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 2):
        super().__init__()
        self.has_residual = (in_channels == out_channels and stride == 1)

        # Depthwise convolution
        self.dw_conv = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=3, stride=stride, padding=1,
            groups=in_channels, bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_channels)

        # Pointwise convolution
        self.pw_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.act = nn.GELU()

        # Residual projection if dimensions change
        if not self.has_residual and stride != 1:
            self.residual_proj = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        elif not self.has_residual:
            self.residual_proj = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.residual_proj = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C_in, H, W)
        Returns:
            (B, C_out, H/stride, W/stride)
        """
        residual = self.residual_proj(x)

        x = self.dw_conv(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.pw_conv(x)
        x = self.bn2(x)
        x = self.act(x)

        return x + residual


class TransformerEncoderLayer(nn.Module):
    """
    Standard Transformer encoder layer: LN -> MHSA -> residual -> LN -> FFN -> residual
    """

    def __init__(self, embed_dim: int = 384, num_heads: int = 6,
                 ffn_expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = FeedForwardNetwork(embed_dim, ffn_expansion, dropout)
        self.drop_path = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, D)
        Returns:
            (B, N, D)
        """
        # MHSA with pre-norm
        x = x + self.drop_path(self.attn(self.norm1(x)))
        # FFN with pre-norm
        x = x + self.drop_path(self.ffn(self.norm2(x)))
        return x


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with scaled dot-product."""

    def __init__(self, embed_dim: int = 384, num_heads: int = 6, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, D)
        Returns:
            (B, N, D)
        """
        B, N, D = x.shape

        # (B, N, 3*D) -> 3 x (B, heads, N, head_dim)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        # Scaled dot-product attention
        # (B, heads, N, head_dim) @ (B, heads, head_dim, N) -> (B, heads, N, N)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # (B, heads, N, N) @ (B, heads, N, head_dim) -> (B, heads, N, head_dim)
        x = (attn @ v).transpose(1, 2).reshape(B, N, D)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class FeedForwardNetwork(nn.Module):
    """FFN with GELU activation: Linear -> GELU -> Dropout -> Linear -> Dropout"""

    def __init__(self, embed_dim: int = 384, expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        hidden_dim = embed_dim * expansion
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, N, D) -> (B, N, D)"""
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention fusion between CNN and Transformer branches.
    CNN features (as queries) attend to Transformer tokens (keys/values).
    Result is added back to Transformer tokens.
    """

    def __init__(self, cnn_dim: int, trans_dim: int, num_heads: int = 6):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = trans_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # CNN features -> queries
        self.q_proj = nn.Linear(trans_dim, trans_dim)
        # Transformer tokens -> keys and values
        self.k_proj = nn.Linear(trans_dim, trans_dim)
        self.v_proj = nn.Linear(trans_dim, trans_dim)

        self.out_proj = nn.Linear(trans_dim, trans_dim)
        self.norm = nn.LayerNorm(trans_dim)

    def forward(self, cnn_feat: torch.Tensor, trans_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cnn_feat: (B, embed_dim, H', W') - projected CNN features
            trans_tokens: (B, N, embed_dim) - Transformer token sequence

        Returns:
            Updated trans_tokens: (B, N, embed_dim)
        """
        B, D, H, W = cnn_feat.shape

        # Flatten CNN spatial dims to sequence: (B, D, H, W) -> (B, H*W, D)
        cnn_seq = rearrange(cnn_feat, "b d h w -> b (h w) d")

        # Queries from CNN, Keys/Values from Transformer
        q = self.q_proj(cnn_seq)  # (B, HW, D)
        k = self.k_proj(trans_tokens)  # (B, N, D)
        v = self.v_proj(trans_tokens)  # (B, N, D)

        # Reshape for multi-head attention
        q = rearrange(q, "b n (h d) -> b h n d", h=self.num_heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.num_heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.num_heads)

        # Cross-attention: CNN queries attend to Transformer keys
        # (B, h, HW, d) @ (B, h, d, N) -> (B, h, HW, N)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        # (B, h, HW, N) @ (B, h, N, d) -> (B, h, HW, d)
        fused = attn @ v
        fused = rearrange(fused, "b h n d -> b n (h d)")
        fused = self.out_proj(fused)

        # Interpolate fused features back to trans_tokens sequence length if needed
        N_trans = trans_tokens.shape[1]
        N_cnn = fused.shape[1]
        if N_cnn != N_trans:
            # Adaptive pooling along sequence dimension
            fused = F.adaptive_avg_pool1d(
                fused.transpose(1, 2), N_trans
            ).transpose(1, 2)

        # Residual addition to Transformer tokens
        trans_tokens = self.norm(trans_tokens + fused)
        return trans_tokens
