"""
Multi-Scale Retinex (MSR) Algorithm.

Mathematical Foundation:
    A single scale sigma cannot simultaneously provide both dynamic range compression 
    and natural color rendition. MSR resolves this compromise by taking a weighted 
    linear sum of multiple Single-Scale Retinex outputs computed across distinct 
    spatial scales (small, medium, large):

        R_{MSR, i}(x, y) = sum_{k=1}^K w_k · [ log(S_i(x, y) + eps) - log(S_i(x, y) * G_{sigma_k}(x, y) + eps) ]

    Where typically:
        K = 3
        weights w = [1/3, 1/3, 1/3]
        sigmas sigma = [15.0, 80.0, 250.0]

Reference:
    Jobson, D. J., Rahman, Z., & Woodell, G. A. (1997).
    "A multiscale retinex for bridging the gap between color images and the human observation of scenes."
    IEEE Transactions on Image Processing, 6(7), 965-976.

Author: Akash Kumar Mishra (Roll No: 24CS003388)
"""

from typing import List, Tuple
import cv2
import numpy as np


def multi_scale_retinex(
    img: np.ndarray,
    sigmas: List[float] = [15.0, 80.0, 250.0],
    weights: List[float] = None,
    eps: float = 1e-4
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Execute Multi-Scale Retinex enhancement on an RGB image.

    Args:
        img: Input image array in range [0.0, 1.0] with shape (H, W, 3).
        sigmas: List of Gaussian standard deviations (scales) in ascending order.
        weights: List of weighting coefficients summing to 1.0. If None, equal weights are assigned.
        eps: Small positive constant added to prevent log(0) singularity.

    Returns:
        Tuple of:
            - enhanced (np.ndarray): Enhanced RGB image in [0.0, 1.0].
            - reflectance (np.ndarray): Multi-scale combined reflectance in [0.0, 1.0].
            - illumination (np.ndarray): Weighted composite illumination field in [0.0, 1.0].
    """
    img_float = np.clip(img.astype(np.float32), 0.0, 1.0)
    h, w, c = img_float.shape

    if weights is None:
        weights = [1.0 / len(sigmas)] * len(sigmas)
    else:
        norm_sum = sum(weights)
        weights = [w / norm_sum for w in weights]

    msr_log = np.zeros_like(img_float)
    composite_illum = np.zeros_like(img_float)

    # Accumulate SSR contributions across all scales
    for sigma, weight in zip(sigmas, weights):
        for ch in range(c):
            channel_data = img_float[:, :, ch]
            illum_k = cv2.GaussianBlur(channel_data, (0, 0), sigmaX=sigma, sigmaY=sigma)
            illum_k = np.maximum(illum_k, eps)
            
            log_s = np.log(channel_data + eps)
            log_i = np.log(illum_k + eps)
            
            msr_log[:, :, ch] += weight * (log_s - log_i)
            composite_illum[:, :, ch] += weight * illum_k

    # Normalization via dynamic range standard deviation stretching
    enhanced = np.zeros_like(msr_log)
    for ch in range(c):
        r_ch = msr_log[:, :, ch]
        mean_val = np.mean(r_ch)
        std_val = np.std(r_ch)
        
        min_val = mean_val - 2.0 * std_val
        max_val = mean_val + 2.0 * std_val
        
        if max_val > min_val:
            stretched = (r_ch - min_val) / (max_val - min_val)
        else:
            stretched = r_ch
            
        enhanced[:, :, ch] = np.clip(stretched, 0.0, 1.0)

    # Reconstruct reflectance in linear domain
    reflectance = np.exp(msr_log)
    for ch in range(c):
        r_linear = reflectance[:, :, ch]
        r_min, r_max = np.percentile(r_linear, 1.0), np.percentile(r_linear, 99.0)
        if r_max > r_min:
            reflectance[:, :, ch] = np.clip((r_linear - r_min) / (r_max - r_min), 0.0, 1.0)
        else:
            reflectance[:, :, ch] = np.clip(r_linear, 0.0, 1.0)

    return enhanced, reflectance, composite_illum
