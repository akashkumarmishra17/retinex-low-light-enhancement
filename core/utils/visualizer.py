"""
Visualization Utilities for Retinex Diagnostics.

Provides tools to generate illumination heatmaps, reflectance maps,
luminance histograms, and visual comparison side-by-sides.

Retinex Project - Akash Kumar Mishra (24CS003388)
"""

from typing import Tuple
import cv2
import numpy as np


def generate_illumination_heatmap(illumination_map: np.ndarray, colormap: int = cv2.COLORMAP_INFERNO) -> np.ndarray:
    """
    Convert a 2D or 3D single-channel illumination map into a vivid false-color heatmap.

    Args:
        illumination_map: Illumination array of shape (H, W) or (H, W, 1) with values in [0, 1].
        colormap: OpenCV colormap constant (default: cv2.COLORMAP_INFERNO).

    Returns:
        np.ndarray: RGB false-color heatmap image in [0, 1].
    """
    if illumination_map.ndim == 3:
        if illumination_map.shape[2] == 3:
            # Convert RGB illumination to grayscale luminance
            gray_illum = cv2.cvtColor(illumination_map, cv2.COLOR_RGB2GRAY)
        else:
            gray_illum = illumination_map.squeeze(-1)
    else:
        gray_illum = illumination_map

    # Normalize to [0, 255] uint8
    norm_illum = np.clip(gray_illum, 0.0, 1.0)
    uint8_illum = (norm_illum * 255.0).astype(np.uint8)

    # Apply colormap (OpenCV returns BGR)
    heatmap_bgr = cv2.applyColorMap(uint8_illum, colormap)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    
    return heatmap_rgb.astype(np.float32) / 255.0


def compute_luminance_histogram(image: np.ndarray, bins: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute normalized luminance histogram of an RGB image.

    Args:
        image: RGB float32 array in [0, 1] or uint8 in [0, 255].
        bins: Number of histogram bins.

    Returns:
        Tuple of (hist_normalized, bin_edges).
    """
    if image.dtype != np.float32 and image.dtype != np.float64:
        img_float = image.astype(np.float32) / 255.0
    else:
        img_float = image

    # Relative luminance via standard Rec.709 coefficients: Y = 0.2126 R + 0.7152 G + 0.0722 B
    luminance = (0.2126 * img_float[:, :, 0] + 
                 0.7152 * img_float[:, :, 1] + 
                 0.0722 * img_float[:, :, 2])
    
    hist, bin_edges = np.histogram(luminance.flatten(), bins=bins, range=(0.0, 1.0), density=True)
    return hist, bin_edges


def create_comparison_grid(original: np.ndarray, enhanced: np.ndarray, 
                           reflectance: np.ndarray = None, illumination: np.ndarray = None) -> np.ndarray:
    """
    Assemble an informative 2x2 or 1x2 diagnostic mosaic.
    """
    h, w, c = original.shape
    if reflectance is not None and illumination is not None:
        # Create 2x2 mosaic
        grid = np.zeros((h * 2, w * 2, 3), dtype=np.float32)
        grid[0:h, 0:w] = original
        grid[0:h, w:2*w] = enhanced
        grid[h:2*h, 0:w] = reflectance
        grid[h:2*h, w:2*w] = generate_illumination_heatmap(illumination)
        return grid
    else:
        # 1x2 side-by-side
        return np.hstack([original, enhanced])
