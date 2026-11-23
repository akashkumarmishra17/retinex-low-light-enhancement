"""
Classical and Baseline Image Enhancement Module.

Implements benchmark comparative methods:
1. SSR: Single-Scale Retinex (Jobson et al. 1997)
2. MSR: Multi-Scale Retinex (Jobson et al. 1997)
3. MSRCP: Multi-Scale Retinex with Chromaticity Preservation (Petro et al. 2014)
4. AGCWD: Adaptive Gamma Correction with Weighting Distribution (Huang et al. 2013)
5. CLAHE: Contrast Limited Adaptive Histogram Equalization (Zuiderveld 1994)
6. MSRCR: Multi-Scale Retinex with Color Restoration (Jobson et al. 1997)
7. HSV-Retinex: Luminance-Domain Retinex in HSV space

Author: Akash Kumar Mishra (Roll No: 24CS003388)
"""

from .ssr import single_scale_retinex
from .msr import multi_scale_retinex
from .msrcp import multi_scale_retinex_chromaticity_preservation
from .msrcr import multi_scale_retinex_color_restoration
from .hsv_retinex import hsv_luminance_retinex
from .agc import adaptive_gamma_correction
from .clahe import clahe_enhancement

__all__ = [
    "single_scale_retinex",
    "multi_scale_retinex",
    "multi_scale_retinex_chromaticity_preservation",
    "multi_scale_retinex_color_restoration",
    "hsv_luminance_retinex",
    "adaptive_gamma_correction",
    "clahe_enhancement",
]
