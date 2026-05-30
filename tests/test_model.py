"""
Tests for model architecture: shape checks and forward pass.
"""

import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.backbone import build_model
from src.models.layers import (
    PatchEmbedding,
    DWSepConv,
    TransformerEncoderBlock,
    CrossAttentionFusion,
)
from src.models.losses import ArcFaceLoss


@pytest.fixture
def model_cfg():
    return {
        "name": "SpikeConformer",
        "num_classes": 164,
        "embed_dim": 384,
        "num_heads": 6,
        "transformer_depth": 6,
        "cnn_channels": [96, 192, 384],
        "patch_size": 16,
        "img_size": 224,
        "dropout": 0.1,
    }


@pytest.fixture
def model(model_cfg):
    return build_model(model_cfg)


class TestPatchEmbedding:
    def test_output_shape(self):
        pe = PatchEmbedding(img_size=224, patch_size=16, in_channels=3, embed_dim=384)
        x = torch.randn(2, 3, 224, 224)
        out = pe(x)
        # (224/16)^2 = 196 patches
        assert out.shape == (2, 196, 384)

    def test_different_patch_size(self):
        pe = PatchEmbedding(img_size=224, patch_size=8, in_channels=3, embed_dim=256)
        x = torch.randn(1, 3, 224, 224)
        out = pe(x)
        # (224/8)^2 = 784 patches
        assert out.shape == (1, 784, 256)


class TestDWSepConv:
    def test_output_shape(self):
        conv = DWSepConv(96, 192, stride=2)
        x = torch.randn(2, 96, 56, 56)
        out = conv(x)
        assert out.shape == (2, 192, 28, 28)

    def test_same_channels(self):
        conv = DWSepConv(128, 128, stride=1)
        x = torch.randn(1, 128, 32, 32)
        out = conv(x)
        assert out.shape == (1, 128, 32, 32)


class TestTransformerEncoder:
    def test_output_shape(self):
        block = TransformerEncoderBlock(dim=384, num_heads=6, mlp_ratio=4.0)
        x = torch.randn(2, 196, 384)
        out = block(x)
        assert out.shape == (2, 196, 384)

    def test_gradient_flow(self):
        block = TransformerEncoderBlock(dim=384, num_heads=6)
        x = torch.randn(2, 196, 384, requires_grad=True)
        out = block(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None


class TestCrossAttentionFusion:
    def test_output_shape(self):
        fusion = CrossAttentionFusion(dim=384, num_heads=6)
        cnn_feat = torch.randn(2, 196, 384)
        trans_feat = torch.randn(2, 196, 384)
        out = fusion(cnn_feat, trans_feat)
        assert out.shape == (2, 196, 384)


class TestFullModel:
    def test_forward_pass(self, model):
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        assert out.shape == (2, 164)

    def test_embeddings(self, model):
        x = torch.randn(2, 3, 224, 224)
        emb = model.get_embeddings(x)
        assert emb.shape == (2, 384)

    def test_param_count(self, model):
        params = model.get_param_count()
        assert params["total"] > 0
        assert params["trainable"] == params["total"]

    def test_no_nan_output(self, model):
        x = torch.randn(4, 3, 224, 224)
        out = model(x)
        assert not torch.isnan(out).any()


class TestArcFaceLoss:
    def test_output_shape(self):
        loss_fn = ArcFaceLoss(num_classes=164, embed_dim=384)
        embeddings = torch.randn(4, 384)
        labels = torch.randint(0, 164, (4,))
        loss = loss_fn(embeddings, labels)
        assert loss.shape == ()
        assert loss.item() > 0

    def test_gradient(self):
        loss_fn = ArcFaceLoss(num_classes=10, embed_dim=128)
        embeddings = torch.randn(4, 128, requires_grad=True)
        labels = torch.randint(0, 10, (4,))
        loss = loss_fn(embeddings, labels)
        loss.backward()
        assert embeddings.grad is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
