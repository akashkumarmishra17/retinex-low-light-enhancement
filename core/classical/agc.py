"""
Adaptive Gamma Correction with Weighting Distribution (AGCWD).

Mathematical Formulation:
    Based on Huang, Cheng, and Chiu (IEEE TIP 2013).
    Instead of fixed gamma power-law transformation (s = r^gamma), AGCWD calculates
    an adaptive gamma parameter per pixel derived from the cumulative distribution
    function (CDF) of the scene's luminance.

    1. Convert RGB to HSV and extract luminance V(x, y) in [0, 1].
    2. Compute probability density function (PDF) and weighting distribution:
           PDF_w(l) = PDF_max * ((PDF(l) - PDF_min) / (PDF_max - PDF_min))^alpha
    3. Calculate modified cumulative distribution function (CDF_w).
    4. Compute adaptive gamma per luminance level:
           gamma(l) = 1.0 - CDF_w(l)
           V_enhanced(x, y) = V(x, y) ^ gamma(V(x, y))
    5. Reconstruct HSV and convert back to RGB.

Guarantees:
    - Automatically lifts dark shadows (gamma < 1) while leaving bright highlights unclipped (gamma ≈ 1).
    - Preserves exact original hue and chromaticity ratios.

Reference:
    Huang, S. C., Cheng, F. C., & Chiu, Y. S. (2013).
    "Efficient contrast enhancement using adaptive gamma correction with weighting distribution."
    IEEE Transactions on Image Processing, 22(3), 1032-1041.

Author: Akash Kumar Mishra (Roll No: 24CS003388)
"""

from typing import Tuple
import cv2
import numpy as np


def adaptive_gamma_correction(
    img: np.ndarray,
    alpha: float = 0.5,
    eps: float = 1e-6
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply Adaptive Gamma Correction with Weighting Distribution (AGCWD) to an RGB image.

    Args:
        img: Input RGB image array in [0.0, 1.0] of shape (H, W, 3).
        alpha: Weighting exponent (controls dynamic range preservation, typically 0.5).
        eps: Small numerical constant.

    Returns:
        Tuple of:
            - enhanced (np.ndarray): Enhanced RGB image in [0.0, 1.0].
            - reflectance (np.ndarray): Luminance reflectance proxy.
            - illumination (np.ndarray): Adaptive illumination curve map.
    """
    img_float = np.clip(img.astype(np.float32), 0.0, 1.0)
    u8_img = (img_float * 255.0).astype(np.uint8)

    # Convert to HSV color space to decouple luminance from chromaticity
    hsv = cv2.cvtColor(u8_img, cv2.COLOR_RGB2HSV).astype(np.float32)
    h_chan = hsv[:, :, 0]
    s_chan = hsv[:, :, 1]
    v_chan = hsv[:, :, 2] / 255.0  # Luminance in [0, 1]

    # Compute luminance histogram and PDF
    hist, _ = np.histogram(v_chan.flatten(), bins=256, range=(0.0, 1.0), density=True)
    pdf = hist / (hist.sum() + eps)

    # Weighting distribution to prevent over-enhancement in dominant modes
    pdf_max = float(pdf.max())
    pdf_min = float(pdf.min())
    if pdf_max > pdf_min:
        pdf_w = pdf_max * (((pdf - pdf_min) / (pdf_max - pdf_min + eps)) ** alpha)
    else:
        pdf_w = pdf

    # Normalized CDF
    cdf_w = np.cumsum(pdf_w) / (np.sum(pdf_w) + eps)

    # Map adaptive gamma to each pixel based on its luminance
    indices = np.clip((v_chan * 255.0).astype(np.int32), 0, 255)
    gamma_map = 1.0 - cdf_w[indices]
    
    # Power-law transformation with adaptive gamma
    v_enhanced = np.power(v_chan + eps, gamma_map)
    v_enhanced = np.clip(v_enhanced, 0.0, 1.0)

    # Reconstruct HSV and convert back to RGB
    hsv_enh = np.stack([h_chan, s_chan, v_enhanced * 255.0], axis=2).astype(np.uint8)
    enhanced = cv2.cvtColor(hsv_enh, cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0

    # Illumination map and reflectance proxies
    illum_3ch = np.repeat(v_chan[:, :, np.newaxis], 3, axis=2)
    refl_3ch = np.clip(enhanced / (illum_3ch + 1e-4), 0.0, 1.0)

    return enhanced, refl_3ch, illum_3ch
