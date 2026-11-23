"""
Full-Reference Image Quality Assessment (IQA).

Used when paired ground-truth normal-light images exist (e.g. LOL Dataset).
Implements:
1. PSNR: Peak Signal-to-Noise Ratio (dB)
2. SSIM: Structural Similarity Index Measure (Wang et al. 2004)
3. MSE: Mean Squared Error
4. MAE: Mean Absolute Error (L1)

Author: Akash Kumar Mishra (Roll No: 24CS003388)
"""

from typing import Dict, Union
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def calculate_psnr(
    enhanced: np.ndarray, 
    reference: np.ndarray, 
    data_range: float = 1.0
) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR) in decibels (dB).

    Formula:
        MSE = (1 / (H * W * C)) * sum( (enhanced - reference)^2 )
        PSNR = 10 * log10( MAX_I^2 / MSE )

    Args:
        enhanced: Enhanced image array in [0.0, 1.0].
        reference: Ground-truth reference image in [0.0, 1.0].
        data_range: Dynamic range of input images (1.0 for float, 255.0 for uint8).

    Returns:
        float: PSNR in dB (typically 18.0 - 28.0 dB for low-light datasets).
    """
    enh = np.clip(enhanced.astype(np.float32), 0.0, data_range)
    ref = np.clip(reference.astype(np.float32), 0.0, data_range)
    
    mse = np.mean((enh - ref) ** 2)
    if mse <= 1e-10:
        return 100.0  # Perfect reconstruction boundary
    
    return float(10.0 * np.log10((data_range ** 2) / mse))


def calculate_ssim(
    enhanced: np.ndarray, 
    reference: np.ndarray, 
    data_range: float = 1.0
) -> float:
    """
    Calculate Structural Similarity Index Measure (SSIM).

    Evaluates luminance l(x, y), contrast c(x, y), and structural s(x, y) correlation:
        SSIM(x, y) = [l(x, y)^alpha] · [c(x, y)^beta] · [s(x, y)^gamma]

    Args:
        enhanced: Enhanced image array in [0.0, 1.0].
        reference: Ground-truth reference image in [0.0, 1.0].
        data_range: Dynamic range of input images (1.0 for float).

    Returns:
        float: SSIM score in range [0.0, 1.0]. Higher is better.
    """
    enh = np.clip(enhanced.astype(np.float32), 0.0, data_range)
    ref = np.clip(reference.astype(np.float32), 0.0, data_range)
    
    # channel_axis=2 for RGB images
    return float(structural_similarity(
        ref, enh, 
        data_range=data_range, 
        channel_axis=2 if enh.ndim == 3 else None
    ))


def calculate_full_reference_metrics(
    enhanced: np.ndarray, 
    reference: np.ndarray
) -> Dict[str, float]:
    """
    Compute comprehensive suite of full-reference metrics.
    """
    psnr_val = calculate_psnr(enhanced, reference)
    ssim_val = calculate_ssim(enhanced, reference)
    
    diff = enhanced.astype(np.float32) - reference.astype(np.float32)
    mse_val = float(np.mean(diff ** 2))
    mae_val = float(np.mean(np.abs(diff)))

    return {
        "PSNR (dB)": round(psnr_val, 3),
        "SSIM": round(ssim_val, 4),
        "MSE": round(mse_val, 5),
        "MAE (L1)": round(mae_val, 5)
    }
