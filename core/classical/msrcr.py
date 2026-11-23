"""
Multi-Scale Retinex with Color Restoration (MSRCR).

Mathematical Foundation:
    The classic MSR often produces "gray-world" color desaturation because it treats
    color channels independently. MSRCR corrects for this by multiplying the MSR result
    with a channel-dependent Color Restoration Factor (CRF):

        C_i(x, y) = beta · [ log(alpha · S_i(x, y)) - log(sum_{c=1}^3 S_c(x, y)) ]

    The restored multi-scale reflectance is:
        R_{MSRCR, i}(x, y) = C_i(x, y) · R_{MSR, i}(x, y)

    Followed by linear gain and offset stretching:
        R_{final, i}(x, y) = G · [ R_{MSRCR, i}(x, y) - b ]

    Canonical empirical parameters (Jobson et al.):
        alpha = 125.0  (chromaticity scale factor)
        beta  = 46.0   (color restoration strength)
        G     = 5.0    (final linear contrast gain)
        b     = 25.0   (luminance offset)

Reference:
    Jobson, D. J., Rahman, Z., & Woodell, G. A. (1997).
    "A multiscale retinex for bridging the gap between color images and the human observation of scenes."
    IEEE Transactions on Image Processing, 6(7), 965-976.

Author: Akash Kumar Mishra (Roll No: 24CS003388)
"""

from typing import List, Tuple
import cv2
import numpy as np


def multi_scale_retinex_color_restoration(
    img: np.ndarray,
    sigmas: List[float] = [15.0, 80.0, 250.0],
    weights: List[float] = None,
    alpha: float = 125.0,
    beta: float = 46.0,
    gain: float = 5.0,
    offset: float = 25.0,
    eps: float = 1e-4
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Execute Multi-Scale Retinex with Color Restoration (MSRCR) on an RGB image.

    Args:
        img: Input image array in range [0.0, 1.0] with shape (H, W, 3).
        sigmas: List of Gaussian standard deviations (scales).
        weights: List of scale weighting coefficients (default: equal weights).
        alpha: Chromaticity balance scale factor.
        beta: Color restoration strength multiplier.
        gain: Final linear gain adjustment factor.
        offset: Contrast shift offset.
        eps: Small constant to avoid log(0).

    Returns:
        Tuple of:
            - enhanced (np.ndarray): Enhanced color-restored RGB image in [0.0, 1.0].
            - reflectance (np.ndarray): Color-restored reflectance in [0.0, 1.0].
            - illumination (np.ndarray): Estimated illumination field in [0.0, 1.0].
    """
    img_float = np.clip(img.astype(np.float32), 0.0, 1.0)
    h, w, c = img_float.shape

    if weights is None:
        weights = [1.0 / len(sigmas)] * len(sigmas)
    else:
        norm_sum = sum(weights)
        weights = [w / norm_sum for w in weights]

    # Scale to [0, 255] for standard Jobson parameter scaling
    img_255 = img_float * 255.0 + 1.0  # +1 prevents log(0)

    # 1. Compute MSR across scales in [0, 255] domain
    msr_log = np.zeros_like(img_255)
    composite_illum = np.zeros_like(img_float)

    for sigma, weight in zip(sigmas, weights):
        for ch in range(c):
            channel_data = img_255[:, :, ch]
            illum_k = cv2.GaussianBlur(channel_data, (0, 0), sigmaX=sigma, sigmaY=sigma)
            illum_k = np.maximum(illum_k, eps)
            
            msr_log[:, :, ch] += weight * (np.log(channel_data) - np.log(illum_k))
            composite_illum[:, :, ch] += weight * (illum_k / 255.0)

    # 2. Compute Color Restoration Factor (CRF):
    # Sum across all 3 color channels per pixel: S_sum(x, y) = R + G + B
    sum_channels = np.sum(img_255, axis=2, keepdims=True)
    crf = np.zeros_like(img_255)
    for ch in range(c):
        crf[:, :, ch] = beta * (np.log(alpha * img_255[:, :, ch]) - np.log(sum_channels[:, :, 0]))

    # 3. Multiply MSR by CRF: R_msrcr = CRF * MSR
    msrcr = crf * msr_log

    # 4. Gain and Offset Application:
    # R_out = gain * (msrcr - offset)
    # Using joint percentile clipping across all channels to preserve color ratio
    p_low = np.percentile(msrcr, 1.0)
    p_high = np.percentile(msrcr, 99.0)
    
    if p_high > p_low:
        enhanced = np.clip((msrcr - p_low) / (p_high - p_low), 0.0, 1.0)
    else:
        enhanced = np.clip(msrcr, 0.0, 1.0)

    # Reconstructed reflectance
    reflectance = np.clip(enhanced, 0.0, 1.0)
    illumination = np.clip(composite_illum, 0.0, 1.0)

    return enhanced, reflectance, illumination
