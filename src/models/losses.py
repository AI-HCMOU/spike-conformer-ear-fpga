"""
ArcFace Loss for angular-margin-based discriminative embedding learning.
Paper uses s=30, m=0.5.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ArcFaceLoss(nn.Module):
    """
    ArcFace: Additive Angular Margin Loss.

    L = -log( exp(s * cos(theta_yi + m)) / (exp(s * cos(theta_yi + m)) + sum_j exp(s * cos(theta_j))) )

    Args:
        num_classes: Number of identity classes
        embed_dim: Embedding dimension
        scale: Feature scale (s), default 30.0
        margin: Angular margin (m) in radians, default 0.5
    """

    def __init__(self, num_classes: int = 164, embed_dim: int = 384,
                 scale: float = 30.0, margin: float = 0.5):
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.num_classes = num_classes

        # Class weight matrix (normalized during forward)
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embed_dim))
        nn.init.xavier_uniform_(self.weight)

        # Precompute cos/sin of margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        # Threshold for numerical stability
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute ArcFace loss.

        Args:
            embeddings: (B, embed_dim) - L2-normalized feature embeddings
            labels: (B,) - ground truth class indices

        Returns:
            Scalar loss value
        """
        # L2 normalize embeddings and weights
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weight = F.normalize(self.weight, p=2, dim=1)

        # Cosine similarity: (B, num_classes)
        cosine = F.linear(embeddings, weight)
        # Clamp for numerical stability
        cosine = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        # Arc-cosine for target class
        sine = torch.sqrt(1.0 - cosine.pow(2))

        # cos(theta + m) = cos(theta)*cos(m) - sin(theta)*sin(m)
        phi = cosine * self.cos_m - sine * self.sin_m

        # When cos(theta) < cos(pi - m), use cosine - mm for stability
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # One-hot encode labels
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.unsqueeze(1), 1.0)

        # Apply margin only to target class
        logits = torch.where(one_hot.bool(), phi, cosine)
        logits = logits * self.scale

        # Cross-entropy on scaled logits
        loss = F.cross_entropy(logits, labels)
        return loss


class ArcFaceHead(nn.Module):
    """
    Combined embedding projection + ArcFace loss.
    Use this when training with ArcFace: pass embeddings from backbone.
    """

    def __init__(self, embed_dim: int = 384, num_classes: int = 164,
                 scale: float = 30.0, margin: float = 0.5):
        super().__init__()
        self.arcface = ArcFaceLoss(num_classes, embed_dim, scale, margin)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute ArcFace loss from embeddings."""
        return self.arcface(embeddings, labels)
