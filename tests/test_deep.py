"""
Unit Tests for Deep Learning Retinex Pipeline (PyTorch).

Validates:
1. DecomNet tensor dimensions, Sigmoid bounds [0, 1].
2. RelightNet forward pass & multi-scale skip connections.
3. RetinexLoss terms (Recon, Mutual, Smoothness).
4. DeepRetinexPipeline end-to-end inference on synthetic low-light input.
5. Checkpoint serialization and deserialization.

Author: Akash Kumar Mishra (Roll No: 24CS003388)
"""

import os
import sys
import torch
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.utils.image_io import create_synthetic_lowlight_sample
from core.deep.models import DecomNet, RelightNet, RetinexLoss
from core.deep.pipeline import DeepRetinexPipeline


def test_deep_retinex_suite():
    print("=" * 60)
    print("Executing Deep Retinex (PyTorch) Verification Suite...")
    print("=" * 60)

    # 1. Test DecomNet
    decom_net = DecomNet()
    dummy_input = torch.rand(2, 3, 128, 128)  # Batch of 2, 3 channels, 128x128
    R, I = decom_net(dummy_input)

    assert R.shape == (2, 3, 128, 128), f"DecomNet R shape mismatch: {R.shape}"
    assert I.shape == (2, 1, 128, 128), f"DecomNet I shape mismatch: {I.shape}"
    assert R.min() >= 0.0 and R.max() <= 1.0, "Reflectance values outside [0, 1]"
    assert I.min() >= 0.0 and I.max() <= 1.0, "Illumination values outside [0, 1]"
    print("[PASS] DecomNet forward pass & tensor shapes validated.")

    # 2. Test RelightNet
    relight_net = RelightNet()
    I_relit = relight_net(R, I)
    assert I_relit.shape == (2, 1, 128, 128), f"RelightNet shape mismatch: {I_relit.shape}"
    assert I_relit.min() >= 0.0 and I_relit.max() <= 1.0, "Relit illumination outside [0, 1]"
    print("[PASS] RelightNet forward pass & multi-scale skip connections validated.")

    # 3. Test RetinexLoss
    criterion = RetinexLoss()
    dummy_norm = torch.rand(2, 3, 128, 128)
    R_norm, I_norm = decom_net(dummy_norm)
    losses = criterion(dummy_input, R, I, dummy_norm, R_norm, I_norm)

    assert "loss_total" in losses
    assert not torch.isnan(losses["loss_total"]), "NaN in loss calculation"
    assert losses["loss_total"].item() > 0.0, "Loss value must be strictly positive"
    print(f"[PASS] RetinexLoss components computed: Total={losses['loss_total'].item():.4f}, "
          f"Recon={losses['loss_recon_low'].item():.4f}, Smooth={losses['loss_smooth'].item():.4f}")

    # 4. Test End-to-End Pipeline
    pipeline = DeepRetinexPipeline()
    img_low, _ = create_synthetic_lowlight_sample(height=180, width=240)
    enhanced, refl, illum = pipeline.enhance(img_low, denoise_reflectance=True)

    assert enhanced.shape == (180, 240, 3), f"Pipeline output shape mismatch: {enhanced.shape}"
    assert enhanced.min() >= 0.0 and enhanced.max() <= 1.0, "Enhanced image outside [0, 1]"
    assert refl.shape == (180, 240, 3), "Reflectance shape mismatch"
    assert illum.shape in [(180, 240, 1), (180, 240, 3)], f"Illumination shape mismatch: {illum.shape}"
    print(f"[PASS] DeepRetinexPipeline inference successful.")
    print(f"       Mean Input: {img_low.mean():.4f} -> Mean Enhanced: {enhanced.mean():.4f}")

    # 5. Test Checkpoint Saving & Loading
    checkpoint_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "checkpoints"))
    checkpoint_file = os.path.join(checkpoint_dir, "deep_retinex_calibrated.pth")
    pipeline.save_weights(checkpoint_file)
    assert os.path.exists(checkpoint_file), "Failed to save model checkpoint"

    # Reload into fresh pipeline
    reloaded_pipeline = DeepRetinexPipeline(checkpoint_path=checkpoint_file)
    enh_reloaded, _, _ = reloaded_pipeline.enhance(img_low)
    np.testing.assert_allclose(enhanced, enh_reloaded, rtol=1e-5, atol=1e-5)
    print(f"[PASS] Checkpoint serialization and reload verified at: {checkpoint_file}")

    print("=" * 60)
    print("ALL DEEP RETINEX TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    test_deep_retinex_suite()
