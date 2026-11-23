"""
Empirical Benchmark Evaluation on Three Distinct Low-Light Scenes.
Generates genuine quantitative metrics (PSNR, SSIM, Entropy, LOE, Sharpness, Latency)
for the Project Report and PowerPoint presentation.

Author: Akash Kumar Mishra (Roll No: 24CS003388)
"""

import json
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.utils.image_io import load_image, save_image
from core.classical.msrcp import multi_scale_retinex_chromaticity_preservation
from core.classical.hsv_retinex import hsv_luminance_retinex
from core.classical.msrcr import multi_scale_retinex_color_restoration
from core.classical.ssr import single_scale_retinex
from core.classical.msr import multi_scale_retinex
from core.deep.pipeline import DeepRetinexPipeline
from core.evaluation.full_reference import calculate_full_reference_metrics
from core.evaluation.no_reference import calculate_no_reference_metrics


def run_benchmarks():
    samples = [
        ("Scene 1: Macbeth Color Chart", "data/samples/sample1_colorchart_low.png", "data/samples/sample1_colorchart_normal.png"),
        ("Scene 2: Night Campus Architecture", "data/samples/sample2_night_low.png", "data/samples/sample2_night_normal.png"),
        ("Scene 3: Indoor AI & Vision Lab", "data/samples/sample3_indoor_low.png", "data/samples/sample3_indoor_normal.png")
    ]

    algorithms = {
        "MSRCP (Proposed/Color Preserved)": lambda img: multi_scale_retinex_chromaticity_preservation(img),
        "HSV Luminance Retinex": lambda img: hsv_luminance_retinex(img),
        "Multi-Scale Retinex (MSR)": lambda img: multi_scale_retinex(img),
        "MSR with Color Restoration (MSRCR)": lambda img: multi_scale_retinex_color_restoration(img),
        "Deep Retinex Pipeline": lambda img: DeepRetinexPipeline().enhance(img)
    }

    results = []
    output_dir = "data/outputs"
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 75)
    print("RUNNING RIGOROUS QUANTITATIVE BENCHMARK ACROSS 3 LOW-LIGHT SCENES")
    print("=" * 75)

    for scene_name, low_path, norm_path in samples:
        img_low = load_image(low_path)
        img_norm = load_image(norm_path)

        orig_brightness = float(np.mean(img_low))
        orig_entropy = float(calculate_no_reference_metrics(img_low, img_low)["Original Entropy (bits)"])

        print(f"\nEvaluating: {scene_name}")
        print(f"  Input: Mean Luminance = {orig_brightness:.4f}, Entropy = {orig_entropy:.2f} bits")

        for algo_name, algo_fn in algorithms.items():
            t0 = time.perf_counter()
            enhanced, refl, illum = algo_fn(img_low)
            latency_ms = (time.perf_counter() - t0) * 1000.0

            # Save sample enhanced outputs for inspection and report/PPT embedding
            clean_name = algo_name.split()[0].lower()
            scene_id = scene_name.split(":")[0].replace(" ", "_").lower()
            save_image(os.path.join(output_dir, f"{scene_id}_{clean_name}_enhanced.png"), enhanced)

            # Quantitative evaluation
            fr = calculate_full_reference_metrics(enhanced, img_norm)
            nr = calculate_no_reference_metrics(img_low, enhanced)

            enh_brightness = float(np.mean(enhanced))

            entry = {
                "scene": scene_name,
                "algorithm": algo_name,
                "input_brightness": round(orig_brightness, 4),
                "enhanced_brightness": round(enh_brightness, 4),
                "psnr_db": fr["PSNR (dB)"],
                "ssim": fr["SSIM"],
                "entropy_in": orig_entropy,
                "entropy_out": nr["Enhanced Entropy (bits)"],
                "entropy_gain": nr["Entropy Gain (bits)"],
                "loe": nr["Lightness Order Error (LOE)"],
                "sharpness": nr["Average Gradient (Sharpness)"],
                "latency_ms": round(latency_ms, 2)
            }
            results.append(entry)

            print(f"  [{algo_name}] -> PSNR: {entry['psnr_db']} dB | SSIM: {entry['ssim']:.4f} | "
                  f"Entropy: {entry['entropy_out']:.2f} bits (+{entry['entropy_gain']:.2f}) | "
                  f"LOE: {entry['loe']} | Time: {entry['latency_ms']:.1f} ms")

    # Save results to json
    results_path = os.path.join(output_dir, "benchmark_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

    print("\n" + "=" * 75)
    print(f"Benchmark results successfully saved to: {results_path}")
    print("=" * 75)
    return results


if __name__ == "__main__":
    run_benchmarks()
