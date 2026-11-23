"""
Image Input/Output and Preprocessing Utilities.

Provides robust functions for loading, saving, color-space converting, 
and normalizing images across NumPy arrays, OpenCV formats, and PyTorch tensors.

Retinex Project - Akash Kumar Mishra (24CS003388)
"""

import io
from typing import Tuple, Union
import cv2
import numpy as np
from PIL import Image


def load_image(source: Union[str, bytes, io.BytesIO], as_float: bool = True) -> np.ndarray:
    """
    Load an image from a filepath or in-memory byte buffer into RGB format.

    Args:
        source: File path (str) or raw bytes/BytesIO buffer from Streamlit uploader.
        as_float: If True, returns float32 in range [0.0, 1.0].
                  If False, returns uint8 in range [0, 255].

    Returns:
        np.ndarray: RGB image array of shape (H, W, 3).
    """
    if isinstance(source, str):
        # Read from disk via OpenCV (reads as BGR)
        img_bgr = cv2.imread(source, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(f"Failed to load image from path: {source}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    elif isinstance(source, (bytes, bytearray)):
        # Decode bytes from memory buffer
        nparr = np.frombuffer(source, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("Failed to decode image from provided byte stream.")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    elif isinstance(source, io.BytesIO):
        pil_img = Image.open(source).convert("RGB")
        img_rgb = np.array(pil_img)
    else:
        raise TypeError(f"Unsupported image source type: {type(source)}")

    if as_float:
        return img_rgb.astype(np.float32) / 255.0
    return img_rgb.astype(np.uint8)


def save_image(filepath: str, img: np.ndarray) -> None:
    """
    Save an RGB image array to disk in JPG or PNG format.

    Args:
        filepath: Target destination path.
        img: RGB image array (either float32 in [0, 1] or uint8 in [0, 255]).
    """
    uint8_img = to_uint8(img)
    bgr_img = cv2.cvtColor(uint8_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filepath, bgr_img)


def to_uint8(img: np.ndarray) -> np.ndarray:
    """
    Safely clip and map a float or int image array to uint8 [0, 255].

    Args:
        img: Input image array.

    Returns:
        np.ndarray: uint8 array bounded within [0, 255].
    """
    if img.dtype == np.uint8:
        return img
    
    # Clip float values to valid [0.0, 1.0] interval before scaling
    clipped = np.clip(img, 0.0, 1.0)
    return (clipped * 255.0).round().astype(np.uint8)


def to_float32(img: np.ndarray) -> np.ndarray:
    """
    Convert an image array to float32 normalized to [0.0, 1.0].
    """
    if img.dtype == np.float32 or img.dtype == np.float64:
        return np.clip(img.astype(np.float32), 0.0, 1.0)
    return (img.astype(np.float32) / 255.0).clip(0.0, 1.0)


def create_synthetic_lowlight_sample(height: int = 400, width: int = 600) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a physically realistic synthetic low-light / normal-light image pair
    for verification, unit tests, and offline demonstrations.

    Models:
        S_normal: Color chart and geometric shapes under uniform illumination.
        S_low: Severe non-uniform illumination gradient + Poisson/Gaussian sensor noise.

    Returns:
        Tuple of (S_low, S_normal) as float32 arrays in [0.0, 1.0].
    """
    # 1. Create a rich normal-light scene with varied textures & color patches
    img_normal = np.zeros((height, width, 3), dtype=np.float32)
    
    # Background subtle gradient
    x = np.linspace(0.4, 0.8, width)
    y = np.linspace(0.4, 0.8, height)
    xx, yy = np.meshgrid(x, y)
    img_normal[:, :, 0] = xx * 0.9
    img_normal[:, :, 1] = yy * 0.85
    img_normal[:, :, 2] = (xx + yy) * 0.45

    # Draw color patches (Macbeth chart style)
    colors = [
        (0.85, 0.25, 0.20),  # Red
        (0.20, 0.80, 0.25),  # Green
        (0.15, 0.35, 0.90),  # Blue
        (0.95, 0.85, 0.15),  # Yellow
        (0.80, 0.20, 0.75),  # Magenta
        (0.15, 0.85, 0.85),  # Cyan
        (0.95, 0.95, 0.95),  # White
        (0.10, 0.10, 0.10),  # Dark Grey
    ]
    
    patch_w, patch_h = width // 5, height // 4
    for idx, col in enumerate(colors):
        r_idx = idx // 4
        c_idx = idx % 4
        y1, y2 = 40 + r_idx * (patch_h + 20), 40 + (r_idx + 1) * patch_h
        x1, x2 = 30 + c_idx * (patch_w + 15), 30 + (c_idx + 1) * patch_w
        if y2 <= height and x2 <= width:
            img_normal[y1:y2, x1:x2, 0] = col[0]
            img_normal[y1:y2, x1:x2, 1] = col[1]
            img_normal[y1:y2, x1:x2, 2] = col[2]

    # Add text banner
    cv_img = (img_normal * 255).astype(np.uint8)
    cv2.putText(cv_img, "Retinex Benchmark - Akash (24CS003388)", (40, height - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    img_normal = cv_img.astype(np.float32) / 255.0

    # 2. Simulate Low-Light Condition via Illumination degradation:
    # Severe non-uniform vignette light field (bright center, dark periphery)
    cx, cy = width * 0.45, height * 0.45
    grid_y, grid_x = np.ogrid[:height, :width]
    dist_sq = ((grid_x - cx) / (width * 0.5)) ** 2 + ((grid_y - cy) / (height * 0.5)) ** 2
    
    # Illumination field I: severely attenuated (0.04 to 0.22)
    illumination = 0.04 + 0.18 * np.exp(-1.5 * dist_sq)
    illumination = illumination[:, :, np.newaxis]

    # S_low = R * I + Sensor Noise
    np.random.seed(42)
    gaussian_noise = np.random.normal(0, 0.015, (height, width, 3)).astype(np.float32)
    
    img_low = np.clip(img_normal * illumination + gaussian_noise, 0.0, 1.0)
    
    return img_low, img_normal
