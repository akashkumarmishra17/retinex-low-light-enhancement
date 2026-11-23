"""
Hybrid Retinex with Adaptive Illumination Correction & Edge-Preserving Guided Filtering.

Proposed Framework:
    An adaptive, physics-grounded Low-Light Image Enhancement pipeline that dynamically
    tunes its parameters based on scene illumination statistics to achieve:
    1. Zero washed-out haze or milky blacks (deep black preservation).
    2. Zero yellow, purple, or unnatural color shifts (chromaticity constancy).
    3. Zero halo artifacts along high-contrast depth boundaries (via Guided Filtering).
    4. Adaptive noise suppression in dark regions without softening bright details.
    5. Clean highlight preservation with zero overexposure burnout.

Mathematical Architecture:
    1. Scene Characteristic Analysis:
           Y(x, y) = 0.299*R + 0.587*G + 0.114*B
           mu_Y = mean(Y),  P_dark = mean(Y < 0.15)
           gamma_base = clip(0.30 + 0.45 * sqrt(mu_Y), 0.30, 0.75)
    2. Illumination Map Estimation:
           L_0(x, y) = max_{c in {R,G,B}} S_c(x, y)
    3. Edge-Preserving Guided Illumination Smoothing:
           L(x, y) = GuidedFilter(Guide=Y, Src=L_0, r=16, eps=1e-3)
    4. Spatially Adaptive Dynamic Range Illumination Tone-Mapping:
           gamma(x, y) = gamma_base + (1.0 - gamma_base) * (L(x, y) ^ 1.2)
           L*(x, y) = L(x, y) ^ gamma(x, y)
    5. Intrinsic Reflectance Extraction & Shadow-Adaptive Denoising:
           R(x, y) = S(x, y) / (L(x, y) + eps)
           R*(x, y) = w(x,y) * Bilateral(R) + (1 - w(x,y)) * R,  where w(x,y) = 1 - L(x,y)
    6. Physical Synthesis & Chromaticity Preservation:
           S*(x, y) = R*(x, y) ⊙ L*(x, y)
           Natural subtle saturation recovery in CIE Lab space.

Author: Akash Kumar Mishra (Roll No: 24CS003388)
"""

from typing import Dict, Tuple
import cv2
import numpy as np


def fast_guided_filter(
    I_guide: np.ndarray,
    p_src: np.ndarray,
    radius: int = 16,
    eps: float = 1e-3
) -> np.ndarray:
    """
    Fast O(N) Guided Filter implementation using OpenCV box filters.
    
    Reference:
        He, K., Sun, J., & Tang, X. (2013).
        "Guided image filtering." IEEE TPAMI, 35(6), 1397-1409.
    """
    ksize = (2 * radius + 1, 2 * radius + 1)
    
    mean_I = cv2.boxFilter(I_guide, cv2.CV_32F, ksize)
    mean_p = cv2.boxFilter(p_src, cv2.CV_32F, ksize)
    mean_Ip = cv2.boxFilter(I_guide * p_src, cv2.CV_32F, ksize)
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = cv2.boxFilter(I_guide * I_guide, cv2.CV_32F, ksize)
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, cv2.CV_32F, ksize)
    mean_b = cv2.boxFilter(b, cv2.CV_32F, ksize)

    q = mean_a * I_guide + mean_b
    return q


def analyze_scene_characteristics(img_float: np.ndarray) -> Dict[str, float]:
    """
    Extract scene lighting and contrast statistics to steer adaptive parameters.
    """
    Y = 0.299 * img_float[:, :, 0] + 0.587 * img_float[:, :, 1] + 0.114 * img_float[:, :, 2]
    
    mean_lum = float(np.mean(Y))
    std_lum = float(np.std(Y))
    dark_ratio = float(np.mean(Y < 0.15))
    bright_ratio = float(np.mean(Y > 0.60))
    p95 = float(np.percentile(Y, 95))
    p05 = float(np.percentile(Y, 5))

    return {
        "mean_luminance": mean_lum,
        "contrast_std": std_lum,
        "dark_ratio": dark_ratio,
        "bright_ratio": bright_ratio,
        "dynamic_range": p95 - p05
    }


def hybrid_adaptive_retinex(
    img: np.ndarray,
    user_gamma_bias: float = 1.0,
    user_contrast_bias: float = 1.0,
    denoise_enabled: bool = True,
    eps: float = 1e-4
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Execute Hybrid Retinex with Adaptive Illumination Correction.

    Args:
        img: Input RGB image array in [0.0, 1.0] with shape (H, W, 3).
        user_gamma_bias: Multiplier for user fine-tuning (1.0 = automatic optimal).
        user_contrast_bias: Multiplier for contrast preference (1.0 = automatic optimal).
        denoise_enabled: If True, executes shadow-weighted edge-preserving denoising.
        eps: Small numerical constant.

    Returns:
        Tuple of:
            - enhanced (np.ndarray): High-quality enhanced RGB image in [0, 1].
            - reflectance (np.ndarray): Recovered intrinsic physical albedo in [0, 1].
            - illumination (np.ndarray): Refined adaptive illumination field in [0, 1].
            - metadata (dict): Computed scene characteristics and adaptive parameters.
    """
    img_float = np.clip(img.astype(np.float32), 0.0, 1.0)
    h, w, c = img_float.shape

    # 1. Luminance & Scene Characteristic Analysis
    Y = 0.299 * img_float[:, :, 0] + 0.587 * img_float[:, :, 1] + 0.114 * img_float[:, :, 2]
    stats = analyze_scene_characteristics(img_float)
    mu_Y = stats["mean_luminance"]
    dark_ratio = stats["dark_ratio"]

    # Compute optimal adaptive gamma base parameter
    # Darker scenes receive a lower base exponent (stronger shadow lift)
    gamma_base = float(np.clip(0.32 + 0.45 * np.sqrt(mu_Y), 0.32, 0.72))
    gamma_base = float(np.clip(gamma_base / max(user_gamma_bias, 0.2), 0.20, 0.95))

    # 2. Initial Illumination Map (Morphological Color Channel Maximum)
    # L_0(x, y) = max_c S_c(x, y) establishes physical upper bound on light
    L_0 = np.max(img_float, axis=2)

    # 3. Edge-Preserving Guided Illumination Smoothing
    # Uses grayscale Y as guidance image to preserve depth boundaries without halo artifacts
    L = fast_guided_filter(Y, L_0, radius=16, eps=1e-3)
    L = np.clip(L, eps, 1.0)

    # 4. Spatially Adaptive Dynamic Range Illumination Correction
    # Dark areas (L -> 0) get lifted strongly; bright areas (L -> 1) receive near-unity exponent
    gamma_map = gamma_base + (1.0 - gamma_base) * (L ** 1.3)
    L_corrected = np.power(L, gamma_map)

    # Dynamic range normalization with protected black level
    p_low = np.percentile(L_corrected, 0.2)
    p_high = np.percentile(L_corrected, 99.8)
    if p_high > p_low:
        L_norm = np.clip((L_corrected - p_low) / (p_high - p_low + eps), 0.0, 1.0)
    else:
        L_norm = L_corrected

    # 5. Intrinsic Reflectance Extraction: R = S / L
    # Single-channel illumination preserves exact input chromaticity ratios R:G:B
    L_3ch = L[:, :, np.newaxis]
    reflectance = np.clip(img_float / (L_3ch + eps), 0.0, 1.0)

    # 6. Shadow-Adaptive Denoising
    # Real camera sensor noise is concentrated in deep shadows.
    # Denoise strictly in dark regions to leave bright textures perfectly sharp.
    if denoise_enabled:
        r_u8 = (reflectance * 255.0).astype(np.uint8)
        r_denoised = cv2.bilateralFilter(r_u8, d=5, sigmaColor=28, sigmaSpace=28).astype(np.float32) / 255.0
        # Shadow weighting mask: w(x,y) -> 1.0 in shadows, 0.0 in highlights
        shadow_weight = np.clip((1.0 - L_norm[:, :, np.newaxis]) * (0.85 + 0.15 * dark_ratio), 0.0, 1.0)
        R_clean = shadow_weight * r_denoised + (1.0 - shadow_weight) * reflectance
    else:
        R_clean = reflectance

    # 7. Physically Grounded Synthesis: S* = R* ⊙ L*
    enhanced_raw = np.clip(R_clean * L_norm[:, :, np.newaxis], 0.0, 1.0)

    # 8. Contrast and Color Balancing in Perceptual CIE Lab Space
    # Applies subtle saturation recovery based on scene darkness to counteract Purkinje desaturation
    u8_raw = (enhanced_raw * 255.0).astype(np.uint8)
    lab = cv2.cvtColor(u8_raw, cv2.COLOR_RGB2LAB).astype(np.float32)
    
    # Natural chroma scaling without hue shift
    chroma_scale = float(np.clip(1.0 + 0.14 * (1.0 - mu_Y) * user_contrast_bias, 1.0, 1.25))
    lab[:, :, 1] = np.clip(128.0 + (lab[:, :, 1] - 128.0) * chroma_scale, 0, 255)
    lab[:, :, 2] = np.clip(128.0 + (lab[:, :, 2] - 128.0) * chroma_scale, 0, 255)

    # Mild luminance S-curve for deep blacks and crisp contrast
    l_chan = lab[:, :, 0] / 255.0
    l_curve = l_chan ** (1.0 / max(user_contrast_bias, 0.5))
    lab[:, :, 0] = np.clip(l_curve * 255.0, 0, 255)

    enhanced = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0

    # Ensure clean bounds
    enhanced = np.clip(enhanced, 0.0, 1.0)
    illum_final = np.repeat(L_norm[:, :, np.newaxis], 3, axis=2)

    metadata = {
        **stats,
        "adaptive_gamma_base": gamma_base,
        "chroma_scale": chroma_scale,
        "algorithm": "Hybrid Retinex (Proposed)"
    }

    return enhanced, R_clean, illum_final, metadata
