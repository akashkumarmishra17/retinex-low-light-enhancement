"""
Single-Scale Retinex (SSR) Algorithm.

Mathematical Foundation:
    S(x, y) = R(x, y) · I(x, y)
    In logarithmic domain:
    r(x, y) = log(S(x, y)) - log(I(x, y))

SSR approximates the smooth illumination field I(x, y) by convolving the
observed image S(x, y) with a 2D Gaussian low-pass spatial kernel G_sigma(x, y):
    I_approx(x, y) = S(x, y) * G_sigma(x, y)
    r(x, y) = log(S(x, y) + eps) - log(S(x, y) * G_sigma(x, y) + eps)

Reference:
    Jobson, D. J., Rahman, Z., & Woodell, G. A. (1997).
    "Properties and performance of a center/surround retinex."
    IEEE Transactions on Image Processing, 6(3), 451-462.

Author: Akash Kumar Mishra (Roll No: 24CS003388)
"""

from typing import Tuple
import cv2
import numpy as np


def single_scale_retinex(
    img: np.ndarray, 
    sigma: float = 80.0, 
    eps: float = 1e-4
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Execute Single-Scale Retinex enhancement on an RGB image.

    Args:
        img: Input image array in range [0.0, 1.0] with shape (H, W, 3).
        sigma: Standard deviation of the Gaussian surround kernel (controls blur radius).
               Smaller sigma (e.g. 15) -> stronger dynamic range compression, more halos.
               Larger sigma (e.g. 250) -> weaker dynamic range compression, better color rendition.
        eps: Small positive constant added to prevent log(0) singularity.

    Returns:
        Tuple of:
            - enhanced (np.ndarray): Enhanced RGB image in [0.0, 1.0].
            - reflectance (np.ndarray): Recovered Reflectance R(x, y) in [0.0, 1.0].
            - illumination (np.ndarray): Estimated Illumination field I(x, y) in [0.0, 1.0].
    """
    img_float = np.clip(img.astype(np.float32), 0.0, 1.0)
    h, w, c = img_float.shape

    # Allocate containers for reflectance and illumination
    illumination = np.zeros_like(img_float)
    retinex_log = np.zeros_like(img_float)

    # Compute Gaussian blur and logarithmic reflectance per color channel
    for ch in range(c):
        channel_data = img_float[:, :, ch]
        
        # Approximate illumination field I_ch via Gaussian convolution
        # kernel size is determined dynamically from sigma: ksize = 0 lets OpenCV compute from sigma
        illum_ch = cv2.GaussianBlur(channel_data, (0, 0), sigmaX=sigma, sigmaY=sigma)
        # Enforce physical positivity constraint: I(x, y) > 0
        illum_ch = np.maximum(illum_ch, eps)
        illumination[:, :, ch] = illum_ch

        # Retinex logarithm subtraction: r = log(S) - log(I)
        log_s = np.log(channel_data + eps)
        log_i = np.log(illum_ch + eps)
        retinex_log[:, :, ch] = log_s - log_i

    # Dynamic range normalization across the logarithmic reflectance
    # Jobson standard approach: linear stretching based on mean and standard deviation
    # to eliminate low/high outliers and prevent harsh clipping.
    enhanced = np.zeros_like(retinex_log)
    for ch in range(c):
        r_ch = retinex_log[:, :, ch]
        mean_val = np.mean(r_ch)
        std_val = np.std(r_ch)
        
        min_val = mean_val - 2.0 * std_val
        max_val = mean_val + 2.0 * std_val
        
        if max_val > min_val:
            stretched = (r_ch - min_val) / (max_val - min_val)
        else:
            stretched = r_ch
        
        enhanced[:, :, ch] = np.clip(stretched, 0.0, 1.0)

    # Reflectance component in linear domain: R = exp(r)
    reflectance = np.exp(retinex_log)
    for ch in range(c):
        r_linear = reflectance[:, :, ch]
        r_min, r_max = np.percentile(r_linear, 1.0), np.percentile(r_linear, 99.0)
        if r_max > r_min:
            reflectance[:, :, ch] = np.clip((r_linear - r_min) / (r_max - r_min), 0.0, 1.0)
        else:
            reflectance[:, :, ch] = np.clip(r_linear, 0.0, 1.0)

    return enhanced, reflectance, illumination
