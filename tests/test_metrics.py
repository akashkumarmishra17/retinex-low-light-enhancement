"""
Unit Tests for Quantitative Evaluation Metrics.

Validates mathematical correctness, boundary limits, and monotonicity
for PSNR, SSIM, LOE, Entropy, and Average Gradient.

Author: Akash Kumar Mishra (Roll No: 24CS003388)
"""

import os
import sys
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.utils.image_io import create_synthetic_lowlight_sample
from core.evaluation.full_reference import calculate_psnr, calculate_ssim, calculate_full_reference_metrics
from core.evaluation.no_reference import calculate_entropy, calculate_average_gradient, calculate_loe, calculate_no_reference_metrics


def test_metrics_suite():
    print("=" * 60)
    print("Executing Evaluation Metrics Verification Suite...")
    print("=" * 60)

    # 1. Test Identity Property on Reference Image
    _, img_normal = create_synthetic_lowlight_sample(height=120, width=160)
    
    psnr_self = calculate_psnr(img_normal, img_normal)
    ssim_self = calculate_ssim(img_normal, img_normal)
    
    assert psnr_self > 80.0, f"Expected PSNR -> inf for identical images, got {psnr_self}"
    assert abs(ssim_self - 1.0) < 1e-4, f"Expected SSIM == 1.0 for identical images, got {ssim_self}"
    print(f"[PASS] Full-Reference Identity property verified (SSIM={ssim_self:.4f}, PSNR={psnr_self:.1f} dB).")

    # 2. Test Perturbation Sensitivity
    noisy = np.clip(img_normal + np.random.normal(0, 0.02, img_normal.shape).astype(np.float32), 0.0, 1.0)
    psnr_noisy = calculate_psnr(noisy, img_normal)
    ssim_noisy = calculate_ssim(noisy, img_normal)

    assert 20.0 < psnr_noisy < 45.0, f"Unexpected PSNR for noisy image: {psnr_noisy}"
    assert 0.60 < ssim_noisy < 0.99, f"Unexpected SSIM for noisy image: {ssim_noisy}"
    print(f"[PASS] Perturbation sensitivity verified (Noisy PSNR={psnr_noisy:.2f} dB, SSIM={ssim_noisy:.4f}).")

    # 3. Test Full-Reference Dict
    metrics_fr = calculate_full_reference_metrics(noisy, img_normal)
    assert "PSNR (dB)" in metrics_fr and "SSIM" in metrics_fr
    assert metrics_fr["MSE"] > 0.0 and metrics_fr["MAE (L1)"] > 0.0
    print("[PASS] calculate_full_reference_metrics dictionary verified.")

    # 4. Test No-Reference Metrics
    img_low, _ = create_synthetic_lowlight_sample(height=120, width=160)
    entropy_low = calculate_entropy(img_low)
    entropy_norm = calculate_entropy(img_normal)
    assert entropy_norm > entropy_low, "Normal image should have higher entropy than low-light image"
    print(f"[PASS] Entropy monotonicity verified (Low: {entropy_low:.2f} bits -> Normal: {entropy_norm:.2f} bits).")

    ag_val = calculate_average_gradient(img_normal)
    assert ag_val > 0.0, "Average gradient must be positive for non-flat images"
    print(f"[PASS] Average gradient verified (AG={ag_val:.4f}).")

    # 5. Test Lightness Order Error (LOE)
    loe_self = calculate_loe(img_normal, img_normal)
    assert loe_self == 0.0, f"LOE between identical images must be 0.0, got {loe_self}"
    
    loe_low = calculate_loe(img_low, img_normal)
    assert loe_low >= 0.0, "LOE must be non-negative"
    print(f"[PASS] Lightness Order Error verified (Self LOE={loe_self:.1f}, Low-Light LOE={loe_low:.2f}).")

    # 6. Test No-Reference Dict
    metrics_nr = calculate_no_reference_metrics(img_low, img_normal)
    assert "Entropy Gain (bits)" in metrics_nr
    assert "Lightness Order Error (LOE)" in metrics_nr
    print("[PASS] calculate_no_reference_metrics dictionary verified.")

    print("=" * 60)
    print("ALL METRICS TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    test_metrics_suite()
