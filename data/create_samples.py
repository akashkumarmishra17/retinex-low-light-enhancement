"""
Generates benchmark sample images in data/samples/ for instant testing.

Author: Akash Kumar Mishra (Roll No: 24CS003388)
"""

import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.utils.image_io import create_synthetic_lowlight_sample, save_image


def generate_benchmark_samples():
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "samples"))
    os.makedirs(output_dir, exist_ok=True)

    # Sample 1: Geometric Color Chart & Text Banner
    s1_low, s1_norm = create_synthetic_lowlight_sample(height=480, width=640)
    save_image(os.path.join(output_dir, "sample1_colorchart_low.png"), s1_low)
    save_image(os.path.join(output_dir, "sample1_colorchart_normal.png"), s1_norm)

    # Sample 2: Architectural Night Scene Simulation
    h, w = 480, 640
    night_scene = np.zeros((h, w, 3), dtype=np.float32)
    # Sky gradient
    for y in range(h // 2):
        night_scene[y, :] = [0.05 + 0.15 * (y / (h/2)), 0.08 + 0.18 * (y / (h/2)), 0.15 + 0.25 * (y / (h/2))]
    # Buildings
    buildings = [(50, 200, 180), (190, 320, 260), (310, 420, 210), (410, 580, 290)]
    for x1, x2, b_h in buildings:
        night_scene[h - b_h:h, x1:x2] = [0.18, 0.20, 0.22]
        # Lit windows
        for wy in range(h - b_h + 20, h - 30, 30):
            for wx in range(x1 + 15, x2 - 15, 25):
                if (wx + wy) % 7 != 0:
                    night_scene[wy:wy+15, wx:wx+12] = [0.95, 0.85, 0.35] # Warm light
    # Road / ground
    night_scene[h // 2:h, :] += [0.08, 0.08, 0.09]
    night_uint8 = (np.clip(night_scene, 0.0, 1.0) * 255.0).astype(np.uint8)
    cv2.putText(night_uint8, "Night Campus Architecture - 24CS003388", (30, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
    night_scene = night_uint8.astype(np.float32) / 255.0
    
    # Degrade to severe low light
    vignette = np.zeros((h, w), dtype=np.float32)
    cv2.circle(vignette, (w // 2, h // 2), int(w * 0.4), 1.0, -1)
    vignette = cv2.GaussianBlur(vignette, (151, 151), 70.0)
    illum = np.clip(0.06 + 0.15 * vignette[:, :, None], 0.0, 1.0)
    noise = np.random.normal(0, 0.012, (h, w, 3)).astype(np.float32)
    s2_low = np.clip(night_scene * illum + noise, 0.0, 1.0)
    
    save_image(os.path.join(output_dir, "sample2_night_low.png"), s2_low)
    save_image(os.path.join(output_dir, "sample2_night_normal.png"), night_scene)

    print(f"[SUCCESS] Sample benchmark images created at: {output_dir}")


if __name__ == "__main__":
    generate_benchmark_samples()
