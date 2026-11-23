"""
No-Reference / Blind Image Quality Assessment (NR-IQA).

Used for real-world low-light photos where no ground truth normal-light image exists.
Implements:
1. Mean Luminance & RMS Contrast.
2. LOE: Lightness Order Error (Wang et al. 2013) - natural illumination order preservation.
3. Entropy: Shannon Information Entropy - dynamic range and textural richness.
4. Average Gradient (AG): Edge sharpness and high-frequency textural recovery.
5. Naturalness Image Quality Index (NSS / MSCN based): Statistical deviation from natural scene distribution.

Author: Akash Kumar Mishra (Roll No: 24CS003388)
"""

from typing import Dict
import cv2
import numpy as np


def calculate_mean_brightness(image: np.ndarray) -> float:
    """Calculate mean luminance of an image in [0.0, 1.0]."""
    img_f = np.clip(image.astype(np.float32), 0.0, 1.0)
    if img_f.ndim == 3:
        lum = 0.299 * img_f[:, :, 0] + 0.587 * img_f[:, :, 1] + 0.114 * img_f[:, :, 2]
    else:
        lum = img_f
    return float(np.mean(lum))


def calculate_contrast(image: np.ndarray) -> float:
    """
    Calculate Root Mean Square (RMS) Contrast.
    RMS contrast is the standard deviation of pixel intensities:
        C_rms = sqrt( 1/N * sum (I(x,y) - I_mean)^2 )
    """
    img_f = np.clip(image.astype(np.float32), 0.0, 1.0)
    if img_f.ndim == 3:
        lum = 0.299 * img_f[:, :, 0] + 0.587 * img_f[:, :, 1] + 0.114 * img_f[:, :, 2]
    else:
        lum = img_f
    return float(np.std(lum))


def calculate_entropy(image: np.ndarray) -> float:
    """
    Calculate Shannon Information Entropy in bits.
    
    Formula:
        H = - sum( p_i * log2(p_i) )
    
    Higher entropy indicates richer information content and expanded dynamic range.
    Low-light images often exhibit low entropy (e.g. 3.5 - 5.0 bits), which increases
    to 7.0 - 7.8 bits after effective Retinex enhancement.
    """
    if image.dtype == np.float32 or image.dtype == np.float64:
        uint8_img = (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    else:
        uint8_img = image

    if uint8_img.ndim == 3:
        gray = cv2.cvtColor(uint8_img, cv2.COLOR_RGB2GRAY)
    else:
        gray = uint8_img

    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    prob = hist / (hist.sum() + 1e-12)
    prob = prob[prob > 0]
    entropy = -np.sum(prob * np.log2(prob))
    return float(entropy)


def calculate_average_gradient(image: np.ndarray) -> float:
    """
    Calculate Average Gradient (AG) to evaluate edge sharpness and detail clarity.
    
    Formula:
        AG = (1 / ((M-1)*(N-1))) * sum( sqrt( (dx^2 + dy^2) / 2 ) )
    """
    if image.dtype != np.float32:
        img_float = image.astype(np.float32) / 255.0
    else:
        img_float = np.clip(image, 0.0, 1.0)

    if img_float.ndim == 3:
        gray = cv2.cvtColor(img_float, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_float

    dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    
    mag = np.sqrt((dx ** 2 + dy ** 2) / 2.0)
    return float(np.mean(mag))


def calculate_loe(original: np.ndarray, enhanced: np.ndarray, downsample_size: int = 64) -> float:
    """
    Calculate Lightness Order Error (LOE).
    
    Reference:
        Wang, S., Zheng, J., Hu, H. M., & Li, B. (2013).
        "Naturalness preserved enhancement algorithm for non-uniform illumination images."
        IEEE Transactions on Image Processing, 22(9), 3538-3548.
        
    Measures how well the relative brightness relationships between image pixels
    are preserved after enhancement.
    Lower LOE indicates superior naturalness and absence of unnatural illumination inversions.
    """
    def get_max_channel(img: np.ndarray) -> np.ndarray:
        if img.ndim == 3:
            return np.max(img, axis=2)
        return img

    l_orig = get_max_channel(original)
    l_enh = get_max_channel(enhanced)

    l_orig_sub = cv2.resize(l_orig.astype(np.float32), (downsample_size, downsample_size), interpolation=cv2.INTER_AREA)
    l_enh_sub = cv2.resize(l_enh.astype(np.float32), (downsample_size, downsample_size), interpolation=cv2.INTER_AREA)

    flat_orig = l_orig_sub.flatten()
    flat_enh = l_enh_sub.flatten()
    n_pixels = len(flat_orig)

    orig_order = flat_orig[:, None] >= flat_orig[None, :]
    enh_order = flat_enh[:, None] >= flat_enh[None, :]

    order_error = np.bitwise_xor(orig_order, enh_order)
    loe_score = float(np.mean(np.sum(order_error, axis=1)) / n_pixels)

    return loe_score


def calculate_naturalness_score(image: np.ndarray) -> float:
    """
    Compute Natural Scene Statistics (NSS) deviation index based on Mean Subtracted
    Contrast Normalized (MSCN) coefficients (Mittal et al. 2013 / NIQE formulation).
    
    Formula:
        MSCN(i, j) = (I(i, j) - mu(i, j)) / (sigma(i, j) + 1.0)
    
    Measures variance and kurtosis deviations from pristine Gaussian distribution.
    Lower score indicates higher naturalness (typical range: 2.5 to 6.0).
    """
    img_f = np.clip(image.astype(np.float32), 0.0, 1.0)
    if img_f.ndim == 3:
        gray = 0.299 * img_f[:, :, 0] + 0.587 * img_f[:, :, 1] + 0.114 * img_f[:, :, 2]
    else:
        gray = img_f

    # Local mean via 7x7 Gaussian window
    mu = cv2.GaussianBlur(gray, (7, 7), 1.166)
    mu_sq = mu * mu
    sigma = np.sqrt(np.abs(cv2.GaussianBlur(gray * gray, (7, 7), 1.166) - mu_sq))

    # MSCN coefficients
    mscn = (gray - mu) / (sigma + 1.0 / 255.0)

    # Compute variance and kurtosis of MSCN
    mscn_var = float(np.var(mscn))
    mscn_kurt = float(np.mean((mscn - np.mean(mscn)) ** 4) / ((mscn_var + 1e-6) ** 2))

    # Analytical Naturalness Index (lower is closer to canonical pristine scene distribution)
    # Pristine natural images exhibit mscn_var ≈ 1.0 and kurtosis ≈ 3.0
    naturalness_index = float(np.abs(mscn_var - 1.0) * 1.5 + np.abs(mscn_kurt - 3.0) * 0.25 + 3.0)
    return round(float(np.clip(naturalness_index, 2.0, 9.0)), 2)


def calculate_no_reference_metrics(original: np.ndarray, enhanced: np.ndarray) -> Dict[str, float]:
    """
    Compute comprehensive suite of no-reference quality metrics.
    """
    b_orig = calculate_mean_brightness(original)
    b_enh = calculate_mean_brightness(enhanced)
    c_orig = calculate_contrast(original)
    c_enh = calculate_contrast(enhanced)
    entropy_orig = calculate_entropy(original)
    entropy_enh = calculate_entropy(enhanced)
    ag_enh = calculate_average_gradient(enhanced)
    loe_val = calculate_loe(original, enhanced)
    niqe_score = calculate_naturalness_score(enhanced)

    return {
        "Mean Brightness": round(b_enh, 4),
        "Brightness Gain": round(b_enh - b_orig, 4),
        "RMS Contrast": round(c_enh, 4),
        "Contrast Gain": round(c_enh - c_orig, 4),
        "Enhanced Entropy (bits)": round(entropy_enh, 3),
        "Original Entropy (bits)": round(entropy_orig, 3),
        "Entropy Gain (bits)": round(entropy_enh - entropy_orig, 3),
        "Average Gradient (Sharpness)": round(ag_enh, 4),
        "Lightness Order Error (LOE)": round(loe_val, 2),
        "Naturalness Score (NIQE-proxy)": niqe_score
    }
