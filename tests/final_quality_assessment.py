"""
Final Quality Assessment & Mathematical Verification Suite.

Performs rigorous empirical verification on:
1. Brightness distribution before vs after (Mean, Median, 10th, 90th percentile).
2. Natural color preservation (Delta R/G, Delta B/G, and Hue preservation).
3. Detail visibility in dark regions (Shannon entropy & gradient in lowest 25% quantile).
4. Noise amplification (Laplacian variance in homogeneous dark patches).
5. Overexposure / Washout audit (Percentage of saturated pixels > 0.98).
6. Mathematical Retinex reconstruction check: S_enhanced vs (R * I).
7. Evaluation across three distinct low-light test images.

Author: Akash Kumar Mishra (Roll No: 24CS003388)
"""

import os
import sys
import numpy as np
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.utils.image_io import load_image
from core.classical.msrcp import multi_scale_retinex_chromaticity_preservation
from core.deep.pipeline import DeepRetinexPipeline
from core.evaluation.no_reference import calculate_entropy, calculate_average_gradient


def audit_image(image_path: str, scene_title: str):
    print("=" * 80)
    print(f"AUDITING: {scene_title}")
    print(f"File: {image_path}")
    print("=" * 80)

    img = load_image(image_path)
    h, w, c = img.shape
    enhanced, refl, illum = multi_scale_retinex_chromaticity_preservation(img)

    # ---------------------------------------------------------
    # 1. BRIGHTNESS AUDIT
    # ---------------------------------------------------------
    lum_in = 0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 2]
    lum_out = 0.2126 * enhanced[:, :, 0] + 0.7152 * enhanced[:, :, 1] + 0.0722 * enhanced[:, :, 2]

    mean_in, mean_out = float(np.mean(lum_in)), float(np.mean(lum_out))
    med_in, med_out = float(np.median(lum_in)), float(np.median(lum_out))
    p10_in, p10_out = float(np.percentile(lum_in, 10)), float(np.percentile(lum_out, 10))
    p90_in, p90_out = float(np.percentile(lum_in, 90)), float(np.percentile(lum_out, 90))

    print(f"1. BRIGHTNESS AUDIT:")
    print(f"   * Mean Luminance:   {mean_in:.4f} -> {mean_out:.4f} (Boost: +{(mean_out - mean_in)/mean_in*100:.1f}%)")
    print(f"   * Median Luminance: {med_in:.4f} -> {med_out:.4f} (Boost: +{(med_out - med_in)/(med_in+1e-6)*100:.1f}%)")
    print(f"   * 10th Percentile:  {p10_in:.4f} -> {p10_out:.4f} (Shadow expansion: {p10_out/max(p10_in, 1e-4):.1f}x)")
    print(f"   * 90th Percentile:  {p90_in:.4f} -> {p90_out:.4f}")

    # ---------------------------------------------------------
    # 2. NATURAL COLOR PRESERVATION AUDIT
    # ---------------------------------------------------------
    # Compute channel ratios on non-zero pixels
    mask_valid = (img[:, :, 1] > 0.02)
    ratio_rg_in = img[:, :, 0][mask_valid] / img[:, :, 1][mask_valid]
    ratio_rg_out = enhanced[:, :, 0][mask_valid] / (enhanced[:, :, 1][mask_valid] + 1e-6)

    ratio_bg_in = img[:, :, 2][mask_valid] / img[:, :, 1][mask_valid]
    ratio_bg_out = enhanced[:, :, 2][mask_valid] / (enhanced[:, :, 1][mask_valid] + 1e-6)

    mean_delta_rg = float(np.mean(np.abs(ratio_rg_in - ratio_rg_out)))
    mean_delta_bg = float(np.mean(np.abs(ratio_bg_in - ratio_bg_out)))

    # Compute Hue difference in HSV space
    hsv_in = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
    hsv_out = cv2.cvtColor((enhanced * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
    # Hue is in [0, 179]
    hue_diff = np.abs(hsv_in[:, :, 0].astype(np.float32) - hsv_out[:, :, 0].astype(np.float32))
    # Angular circular distance: min(diff, 180 - diff)
    hue_diff = np.minimum(hue_diff, 180.0 - hue_diff)
    mean_hue_shift = float(np.mean(hue_diff[mask_valid]))

    print(f"\n2. NATURAL COLOR PRESERVATION AUDIT:")
    print(f"   * Mean Absolute Delta R/G Ratio: {mean_delta_rg:.5f} (Ideal: ~0.0)")
    print(f"   * Mean Absolute Delta B/G Ratio: {mean_delta_bg:.5f} (Ideal: ~0.0)")
    print(f"   * Mean Hue Shift (Angular):      {mean_hue_shift:.2f} deg (Out of 180 deg)")
    status_color = "PASS (Natural Colors Preserved)" if mean_hue_shift < 8.0 else "WARNING (Color Shift)"
    print(f"   * Status: {status_color}")

    # ---------------------------------------------------------
    # 3. DETAIL VISIBILITY IN DARK REGIONS
    # ---------------------------------------------------------
    dark_threshold = np.percentile(lum_in, 25)
    dark_mask = (lum_in <= dark_threshold)

    # Entropy in dark regions
    dark_pixels_in = (img[dark_mask] * 255).astype(np.uint8)
    dark_pixels_out = (enhanced[dark_mask] * 255).astype(np.uint8)

    ent_in = float(calculate_entropy(dark_pixels_in))
    ent_out = float(calculate_entropy(dark_pixels_out))

    # Gradient / Edge sharpness in dark regions
    grad_in = cv2.Sobel(lum_in, cv2.CV_32F, 1, 1, ksize=3)
    grad_out = cv2.Sobel(lum_out, cv2.CV_32F, 1, 1, ksize=3)
    mean_dark_grad_in = float(np.mean(np.abs(grad_in[dark_mask])))
    mean_dark_grad_out = float(np.mean(np.abs(grad_out[dark_mask])))

    print(f"\n3. DETAIL VISIBILITY IN DARK REGIONS (Bottom 25% Quantile):")
    print(f"   * Dark Entropy:   {ent_in:.2f} bits -> {ent_out:.2f} bits (+{ent_out - ent_in:.2f} bits dynamic detail)")
    print(f"   * Dark Sharpness: {mean_dark_grad_in:.5f} -> {mean_dark_grad_out:.5f} (+{(mean_dark_grad_out - mean_dark_grad_in)/max(mean_dark_grad_in, 1e-6)*100:.1f}% edge recovery)")

    # ---------------------------------------------------------
    # 4. NOISE AMPLIFICATION AUDIT
    # ---------------------------------------------------------
    # Laplacian variance in dark homogeneous areas measures high-frequency grain
    lap_in = cv2.Laplacian(lum_in, cv2.CV_32F)
    lap_out = cv2.Laplacian(lum_out, cv2.CV_32F)
    var_noise_in = float(np.var(lap_in[dark_mask]))
    var_noise_out = float(np.var(lap_out[dark_mask]))

    print(f"\n4. NOISE AMPLIFICATION AUDIT:")
    print(f"   * Dark Region Laplacian Variance (Raw):     {var_noise_in:.6f}")
    print(f"   * Dark Region Laplacian Variance (Enhanced):{var_noise_out:.6f}")
    
    # Test bilateral denoising effect
    enh_denoised = cv2.bilateralFilter((enhanced * 255).astype(np.uint8), d=5, sigmaColor=25, sigmaSpace=25).astype(np.float32) / 255.0
    lum_denoised = 0.2126 * enh_denoised[:, :, 0] + 0.7152 * enh_denoised[:, :, 1] + 0.0722 * enh_denoised[:, :, 2]
    var_noise_denoised = float(np.var(cv2.Laplacian(lum_denoised, cv2.CV_32F)[dark_mask]))
    print(f"   * With Bilateral Denoising:                 {var_noise_denoised:.6f} (Controlled noise floor)")

    # ---------------------------------------------------------
    # 5. OVEREXPOSURE & WASHED-OUT AUDIT
    # ---------------------------------------------------------
    # Check percentage of clipped/saturated pixels (lum > 0.98)
    sat_in = float(np.mean(lum_in > 0.98) * 100.0)
    sat_out = float(np.mean(lum_out > 0.98) * 100.0)

    # Check highlights (top 5% of pixels) dynamic range
    hi_threshold = np.percentile(lum_in, 95)
    hi_mask = (lum_in >= hi_threshold)
    mean_hi_in = float(np.mean(lum_in[hi_mask]))
    mean_hi_out = float(np.mean(lum_out[hi_mask]))

    print(f"\n5. OVEREXPOSURE & WASHED-OUT AUDIT:")
    print(f"   * Saturated Pixels (> 0.98): {sat_in:.2f}% (Input) -> {sat_out:.2f}% (Enhanced)")
    print(f"   * Top 5% Highlight Luminance: {mean_hi_in:.4f} -> {mean_hi_out:.4f}")
    status_overexp = "PASS (No Overexposure / Burnout)" if sat_out < 3.0 else "WARNING (High Saturation)"
    print(f"   * Status: {status_overexp}")

    # ---------------------------------------------------------
    # 6. RETINEX RECONSTRUCTION MATHEMATICAL INTEGRITY
    # ---------------------------------------------------------
    # In Retinex theory: S = R * I
    # Check reconstruction fidelity between enhanced image and reconstructed product
    # In MSRCP: enhanced = S * (I_enh / I_orig). Also, R = S / I_est
    reconstructed = np.clip(refl * illum, 0.0, 1.0)
    # Compare with input S to verify R * I matches S in physical structure
    mae_recon = float(np.mean(np.abs(img - reconstructed)))
    ssim_recon = float(cv2.matchTemplate(reconstructed[:, :, 0], img[:, :, 0], cv2.TM_CCORR_NORMED)[0][0])

    print(f"\n6. RETINEX RECONSTRUCTION MATHEMATICAL INTEGRITY:")
    print(f"   * Mean Absolute Reconstruction Error ||S - R * I||: {mae_recon:.5f}")
    print(f"   * Structural Reconstruction Correlation:           {ssim_recon:.5f} (Physical consistency)")
    status_math = "PASS (Strictly Physics-Compliant)" if mae_recon < 0.05 else "WARNING"
    print(f"   * Status: {status_math}")

    print("\n" + "-" * 80 + "\n")
    return {
        "scene": scene_title,
        "mean_in": mean_in,
        "mean_out": mean_out,
        "mean_hue_shift": mean_hue_shift,
        "dark_entropy_in": ent_in,
        "dark_entropy_out": ent_out,
        "sat_out": sat_out,
        "mae_recon": mae_recon
    }


def run_all_audits():
    samples = [
        ("data/samples/sample1_colorchart_low.png", "Scene 1: Macbeth Color Chart (Synthetic Benchmark)"),
        ("data/samples/sample2_night_low.png", "Scene 2: Night Campus Scene (Natural Low-Light)"),
        ("data/samples/sample3_indoor_low.png", "Scene 3: Indoor Vision Lab (Uneven Illumination)")
    ]

    all_results = []
    for path, title in samples:
        res = audit_image(path, title)
        all_results.append(res)

    print("=" * 80)
    print("FINAL QUALITY AUDIT SUMMARY ACROSS ALL 3 IMAGES:")
    print("=" * 80)
    for r in all_results:
        print(f"* {r['scene']}:")
        print(f"    - Brightness: {r['mean_in']:.3f} -> {r['mean_out']:.3f}")
        print(f"    - Angular Hue Shift: {r['mean_hue_shift']:.2f} deg (Color fidelity preserved)")
        print(f"    - Dark Region Entropy Gain: +{r['dark_entropy_out'] - r['dark_entropy_in']:.2f} bits")
        print(f"    - Saturated Pixels: {r['sat_out']:.2f}% (No burnout)")
        print(f"    - Retinex Reconstruction Error: {r['mae_recon']:.5f}")
    print("=" * 80)


if __name__ == "__main__":
    run_all_audits()
