"""
Contrast Limited Adaptive Histogram Equalization (CLAHE) in Perceptual Color Space.

Mathematical Formulation:
    Standard Global Histogram Equalization calculates a single transfer function across
    the entire image, which often over-amplifies noise in flat regions and causes highlight washout.
    
    CLAHE divides the image into contextual tiles (e.g. 8x8) and computes local histogram
    equalization with a strict slope clip-limit:
        - Histogram slope is clipped at 'clipLimit' threshold.
        - Excess probability mass is uniformly redistributed across all bins.
        - Bilinear interpolation is used across tile boundaries to eliminate block artifacts.

    Applied in CIE Lab Color Space:
        - Applied strictly to the L* (Luminance) channel.
        - Chrominance channels a* (green-red) and b* (blue-yellow) are left intact.
        - Guarantees zero chromatic shift.

Reference:
    Zuiderveld, K. (1994).
    "Contrast limited adaptive histogram equalization."
    Graphics Gems IV, Academic Press, pp. 474-485.

Author: Akash Kumar Mishra (Roll No: 24CS003388)
"""

from typing import Tuple
import cv2
import numpy as np


def clahe_enhancement(
    img: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply CLAHE in CIE Lab space strictly to the luminance channel.

    Args:
        img: Input RGB image array in [0.0, 1.0] of shape (H, W, 3).
        clip_limit: Threshold for contrast limiting (typically 2.0 to 3.0).
        tile_grid_size: Number of contextual tiles (default: (8, 8)).

    Returns:
        Tuple of (enhanced, reflectance_proxy, illumination_proxy).
    """
    img_float = np.clip(img.astype(np.float32), 0.0, 1.0)
    u8_img = (img_float * 255.0).astype(np.uint8)

    # Convert to CIE Lab
    lab = cv2.cvtColor(u8_img, cv2.COLOR_RGB2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)

    # Create CLAHE operator and process luminance channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_enhanced = clahe.apply(l_chan)

    # Merge channels and convert back to RGB
    lab_enhanced = cv2.merge([l_enhanced, a_chan, b_chan])
    enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0

    # Illumination and reflectance proxy
    illum_proxy = np.repeat((l_chan.astype(np.float32) / 255.0)[:, :, np.newaxis], 3, axis=2)
    refl_proxy = np.clip(enhanced / (illum_proxy + 1e-4), 0.0, 1.0)

    return enhanced, refl_proxy, illum_proxy
