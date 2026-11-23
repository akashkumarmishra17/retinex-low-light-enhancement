"""
Unit Tests for Classical Retinex Algorithms.

Validates mathematical output bounds, dimensions, and numerical stability
for SSR, MSR, MSRCR, MSRCP, and HSV-Retinex against synthetic test patterns.

Author: Akash Kumar Mishra (Roll No: 24CS003388)
"""

import os
import sys
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.utils.image_io import create_synthetic_lowlight_sample
from core.classical.ssr import single_scale_retinex
from core.classical.msr import multi_scale_retinex
from core.classical.msrcr import multi_scale_retinex_color_restoration
from core.classical.msrcp import multi_scale_retinex_chromaticity_preservation
from core.classical.hsv_retinex import hsv_luminance_retinex
from core.classical.agc import adaptive_gamma_correction
from core.classical.clahe import clahe_enhancement
from core.adaptive.hybrid_retinex import hybrid_adaptive_retinex


def test_classical_retinex_suite():
    print("=" * 60)
    print("Executing Classical Retinex Verification Suite...")
    print("=" * 60)

    # 1. Generate synthetic low-light / normal pair
    img_low, img_normal = create_synthetic_lowlight_sample(height=200, width=300)
    assert img_low.shape == (200, 300, 3), "Invalid synthetic image dimensions"
    assert img_low.min() >= 0.0 and img_low.max() <= 1.0, "Input image not in [0, 1]"
    print("[PASS] Synthetic Low-Light generation validated.")

    # 2. Test Single-Scale Retinex (SSR)
    for test_sigma in [15.0, 80.0, 250.0]:
        enh_ssr, refl_ssr, illum_ssr = single_scale_retinex(img_low, sigma=test_sigma)
        assert enh_ssr.shape == img_low.shape, f"SSR output shape mismatch for sigma={test_sigma}"
        assert not np.isnan(enh_ssr).any(), f"NaN detected in SSR output for sigma={test_sigma}"
        assert enh_ssr.min() >= 0.0 and enh_ssr.max() <= 1.0, "SSR enhanced image outside [0, 1]"
    print("[PASS] Single-Scale Retinex (SSR) validated across scales (15, 80, 250).")

    # 3. Test Multi-Scale Retinex (MSR)
    enh_msr, refl_msr, illum_msr = multi_scale_retinex(img_low)
    assert enh_msr.shape == img_low.shape, "MSR output shape mismatch"
    assert not np.isnan(enh_msr).any(), "NaN detected in MSR output"
    assert enh_msr.min() >= 0.0 and enh_msr.max() <= 1.0, "MSR enhanced image outside [0, 1]"
    print("[PASS] Multi-Scale Retinex (MSR) validated.")

    # 4. Test Multi-Scale Retinex with Color Restoration (MSRCR)
    enh_msrcr, refl_msrcr, illum_msrcr = multi_scale_retinex_color_restoration(img_low)
    assert enh_msrcr.shape == img_low.shape, "MSRCR output shape mismatch"
    assert not np.isnan(enh_msrcr).any(), "NaN detected in MSRCR output"
    assert enh_msrcr.min() >= 0.0 and enh_msrcr.max() <= 1.0, "MSRCR enhanced image outside [0, 1]"
    print("[PASS] Multi-Scale Retinex with Color Restoration (MSRCR) validated.")

    # 5. Test Multi-Scale Retinex with Chromaticity Preservation (MSRCP)
    enh_msrcp, refl_msrcp, illum_msrcp = multi_scale_retinex_chromaticity_preservation(img_low)
    assert enh_msrcp.shape == img_low.shape, "MSRCP output shape mismatch"
    assert not np.isnan(enh_msrcp).any(), "NaN detected in MSRCP output"
    assert enh_msrcp.min() >= 0.0 and enh_msrcp.max() <= 1.0, "MSRCP enhanced image outside [0, 1]"
    
    # Check chromaticity preservation: R, G, B channel means should be balanced
    ch_means = enh_msrcp.mean(axis=(0, 1))
    assert abs(ch_means[0] - ch_means[1]) < 0.1, "MSRCP introduced unexpected channel imbalance"
    print(f"[PASS] MSRCP Chromaticity Preservation validated (Channel Means: {ch_means}).")

    # 6. Test HSV Luminance Retinex
    enh_hsv, refl_hsv, illum_hsv = hsv_luminance_retinex(img_low)
    assert enh_hsv.shape == img_low.shape, "HSV-Retinex output shape mismatch"
    assert not np.isnan(enh_hsv).any(), "NaN in HSV-Retinex"
    assert enh_hsv.min() >= 0.0 and enh_hsv.max() <= 1.0, "HSV-Retinex outside [0, 1]"
    print("[PASS] HSV Luminance Retinex validated.")

    # 7. Test AGCWD
    enh_agcwd, refl_agcwd, illum_agcwd = adaptive_gamma_correction(img_low)
    assert enh_agcwd.shape == img_low.shape, "AGCWD output shape mismatch"
    assert not np.isnan(enh_agcwd).any(), "NaN in AGCWD"
    assert enh_agcwd.min() >= 0.0 and enh_agcwd.max() <= 1.0, "AGCWD outside [0, 1]"
    print("[PASS] AGCWD validated.")

    # 8. Test CLAHE
    enh_clahe, refl_clahe, illum_clahe = clahe_enhancement(img_low)
    assert enh_clahe.shape == img_low.shape, "CLAHE output shape mismatch"
    assert not np.isnan(enh_clahe).any(), "NaN in CLAHE"
    assert enh_clahe.min() >= 0.0 and enh_clahe.max() <= 1.0, "CLAHE outside [0, 1]"
    print("[PASS] CLAHE validated.")

    # 9. Test Proposed Adaptive Hybrid Retinex
    enh_hyb, refl_hyb, illum_hyb, meta_hyb = hybrid_adaptive_retinex(img_low)
    assert enh_hyb.shape == img_low.shape, "Adaptive Hybrid Retinex output shape mismatch"
    assert not np.isnan(enh_hyb).any(), "NaN in Adaptive Hybrid Retinex"
    assert enh_hyb.min() >= 0.0 and enh_hyb.max() <= 1.0, "Adaptive Hybrid Retinex outside [0, 1]"
    assert "adaptive_gamma_base" in meta_hyb, "Missing metadata in Adaptive Hybrid Retinex"
    print("[PASS] Proposed Adaptive Hybrid Retinex validated.")

    print("=" * 60)
    print("ALL CLASSICAL RETINEX TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    test_classical_retinex_suite()
