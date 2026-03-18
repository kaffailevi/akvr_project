"""Tests for EarlyFusion and MidLevelFusion."""

import numpy as np
import pytest
import torch


def _make_rgb_tensor(b: int = 2, h: int = 64, w: int = 64) -> torch.Tensor:
    return torch.rand(b, 3, h, w)


def _make_ir_tensor(b: int = 2, h: int = 64, w: int = 64) -> torch.Tensor:
    return torch.rand(b, 1, h, w)


# ---------------------------------------------------------------------------
# EarlyFusion tests
# ---------------------------------------------------------------------------

def test_early_fusion_shape() -> None:
    """EarlyFusion must produce a (4, H, W) tensor from (3, H, W) + (1, H, W)."""
    from src.fusion.fusion import EarlyFusion

    fuser = EarlyFusion()
    rgb = torch.rand(3, 64, 64)
    ir = torch.rand(1, 64, 64)
    fused = fuser(rgb, ir)

    assert fused.shape == (4, 64, 64), f"Unexpected fused shape: {fused.shape}"


def test_early_fusion_channels() -> None:
    """First 3 channels must be RGB; 4th channel must be IR."""
    from src.fusion.fusion import EarlyFusion

    fuser = EarlyFusion()
    rgb = torch.rand(3, 32, 32)
    ir = torch.rand(1, 32, 32)
    fused = fuser(rgb, ir)

    assert torch.allclose(fused[:3], rgb), "First 3 channels should match RGB"
    assert torch.allclose(fused[3:4], ir), "4th channel should match IR"


def test_early_fusion_invalid_rgb_channels() -> None:
    """EarlyFusion must raise ValueError if RGB tensor doesn't have 3 channels."""
    from src.fusion.fusion import EarlyFusion

    fuser = EarlyFusion()
    with pytest.raises(ValueError):
        fuser(torch.rand(4, 64, 64), torch.rand(1, 64, 64))


def test_early_fusion_invalid_ir_channels() -> None:
    """EarlyFusion must raise ValueError if IR tensor doesn't have 1 channel."""
    from src.fusion.fusion import EarlyFusion

    fuser = EarlyFusion()
    with pytest.raises(ValueError):
        fuser(torch.rand(3, 64, 64), torch.rand(3, 64, 64))


def test_early_fusion_to_tensor() -> None:
    """to_tensor must convert numpy arrays to correctly shaped tensors."""
    from src.fusion.fusion import EarlyFusion

    rgb_np = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    ir_np = np.random.randint(0, 256, (64, 64), dtype=np.uint8)

    rgb_t, ir_t = EarlyFusion.to_tensor(rgb_np, ir_np)

    assert rgb_t.shape == (3, 64, 64), f"Unexpected RGB tensor shape: {rgb_t.shape}"
    assert ir_t.shape == (1, 64, 64), f"Unexpected IR tensor shape: {ir_t.shape}"
    assert rgb_t.dtype == torch.float32
    assert ir_t.dtype == torch.float32
    # Values should be in [0, 1]
    assert float(rgb_t.max()) <= 1.0
    assert float(ir_t.max()) <= 1.0


def test_early_fusion_to_tensor_hwc1() -> None:
    """to_tensor must handle IR images with shape (H, W, 1)."""
    from src.fusion.fusion import EarlyFusion

    rgb_np = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
    ir_np = np.random.randint(0, 256, (32, 32, 1), dtype=np.uint8)

    _, ir_t = EarlyFusion.to_tensor(rgb_np, ir_np)
    assert ir_t.shape == (1, 32, 32)


# ---------------------------------------------------------------------------
# MidLevelFusion tests
# ---------------------------------------------------------------------------

def test_mid_level_fusion_forward() -> None:
    """Forward pass must produce (B, num_classes) logits."""
    from src.fusion.fusion import MidLevelFusion

    num_classes = 5
    model = MidLevelFusion(num_classes=num_classes)
    model.eval()

    rgb = _make_rgb_tensor(b=2)
    ir = _make_ir_tensor(b=2)

    with torch.no_grad():
        out = model(rgb, ir)

    assert out.shape == (2, num_classes), f"Unexpected output shape: {out.shape}"


def test_mid_level_fusion_custom_classes() -> None:
    """MidLevelFusion must support arbitrary num_classes."""
    from src.fusion.fusion import MidLevelFusion

    model = MidLevelFusion(num_classes=10)
    model.eval()

    with torch.no_grad():
        out = model(_make_rgb_tensor(b=1), _make_ir_tensor(b=1))

    assert out.shape == (1, 10)


def test_mid_level_fusion_gradient_flow() -> None:
    """Gradients should flow through the model during a backward pass."""
    from src.fusion.fusion import MidLevelFusion

    model = MidLevelFusion(num_classes=5)

    rgb = _make_rgb_tensor(b=1).requires_grad_(True)
    ir = _make_ir_tensor(b=1).requires_grad_(True)

    out = model(rgb, ir)
    loss = out.sum()
    loss.backward()

    assert rgb.grad is not None, "No gradient flowed back through RGB stream"
    assert ir.grad is not None, "No gradient flowed back through IR stream"


def test_mid_level_fusion_batch_size_one() -> None:
    """MidLevelFusion must handle batch size of 1."""
    from src.fusion.fusion import MidLevelFusion

    model = MidLevelFusion(num_classes=5)
    model.eval()

    with torch.no_grad():
        out = model(_make_rgb_tensor(b=1), _make_ir_tensor(b=1))

    assert out.shape == (1, 5)
