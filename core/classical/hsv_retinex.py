"""
HSV Luminance-Domain Retinex.

Enhances only the Value (V) channel in HSV color space while keeping Hue (H) 
and Saturation (S) completely unaltered. This provides 100% hue constancy.

Author: Akash Kumar Mishra (Roll No: 24CS003388)
"""

from typing import List, Tuple
import cv2
import numpy as np


def hsv_luminance_retinex(
    img: np.ndarray,
    sigmas: List[float] = [15.0, 80.0, 250.0],
    weights: List[float] = None,
    gamma: float = 1.0,
    eps: float = 1e-4
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Execute Multi-Scale Retinex on the V (Luminance) channel of HSV color space.

    Args:
        img: Input RGB image in [0.0, 1.0].
        sigmas: Gaussian blur scales.
        weights: Weights for each scale.
        gamma: Optional gamma adjustment on enhanced luminance.
        eps: Small constant to avoid log(0).

    Returns:
        Tuple of (enhanced_rgb, reflectance, illumination).
    """
    img_float = np.clip(img.astype(np.float32), 0.0, 1.0)
    uint8_img = (img_float * 255.0).astype(np.uint8)

    # Convert to HSV
    hsv = cv2.cvtColor(uint8_img, cv2.COLOR_RGB2HSV).astype(np.float32) / 255.0
    h_chan = hsv[:, :, 0]
    s_chan = hsv[:, :, 1]
    v_chan = hsv[:, :, 2]  # Luminance in [0, 1]

    if weights is None:
        weights = [1.0 / len(sigmas)] * len(sigmas)
    else:
        norm_sum = sum(weights)
        weights = [w / norm_sum for w in weights]

    msr_v = np.zeros_like(v_chan)
    illum_v = np.zeros_like(v_chan)

    for sigma, weight in zip(sigmas, weights):
        illum_k = cv2.GaussianBlur(v_chan, (0, 0), sigmaX=sigma, sigmaY=sigma)
        illum_k = np.maximum(illum_k, eps)

        log_v = np.log(v_chan + eps)
        log_illum = np.log(illum_k + eps)

        msr_v += weight * (log_v - log_illum)
        illum_v += weight * illum_k

    # Dynamic range normalization on V
    p_low, p_high = np.percentile(msr_v, 1.0), np.percentile(msr_v, 99.0)
    if p_high > p_low:
        v_enh = (msr_v - p_low) / (p_high - p_low)
    else:
        v_enh = msr_v

    v_enh = np.clip(v_enh, 0.0, 1.0)
    if gamma != 1.0:
        v_enh = np.power(v_enh, 1.0 / gamma)

    # Slight saturation compensation to prevent desaturation
    s_enh = np.clip(s_chan * 1.1, 0.0, 1.0)

    # Reconstruct HSV and convert back to RGB
    hsv_enh = np.stack([h_chan * 179.0, s_enh * 255.0, v_enh * 255.0], axis=2).astype(np.uint8)
    enhanced_rgb = cv2.cvtColor(hsv_enh, cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0

    # Reflectance and illumination
    illum_3ch = np.repeat(illum_v[:, :, np.newaxis], 3, axis=2)
    reflectance = np.clip(img_float / (illum_3ch + eps), 0.0, 1.0)
    p1, p99 = np.percentile(reflectance, 1.0), np.percentile(reflectance, 99.0)
    reflectance = np.clip((reflectance - p1) / (p99 - p1 + 1e-6), 0.0, 1.0)

    return enhanced_rgb, reflectance, illum_3ch
