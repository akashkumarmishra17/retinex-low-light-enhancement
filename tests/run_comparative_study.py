"""
Comprehensive Comparative Evaluation Suite (6 Algorithms x 5 Low-Light Scenes).

Evaluates:
1. Single-Scale Retinex (SSR)
2. Multi-Scale Retinex (MSR)
3. Multi-Scale Retinex with Colour Preservation (MSRCP)
4. Adaptive Gamma Correction with Weighting Distribution (AGCWD)
5. CLAHE (Contrast Limited Adaptive Histogram Equalization in Lab Space)
6. Hybrid Adaptive Retinex (Proposed Framework)

Across 5 Distinct Low-Light Scenes:
- Scene 1: Macbeth Color Chart (Synthetic Benchmark)
- Scene 2: Night Campus Scene (Natural Low Light)
- Scene 3: Indoor Vision Lab (Uneven Illumination)
- Scene 4: Urban Night Roadway (Deep Shadows & Directional Light)
- Scene 5: Dimly Lit Portrait (Human Subject & Skin Tones)

Saves:
- All enhanced output images
- Full CSV report (benchmark_results.csv)
- High-resolution comparative visualization charts (PNG)

Author: Akash Kumar Mishra (Roll No: 24CS003388)
"""

import os
import sys
import time
import json
import csv
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.utils.image_io import load_image, save_image
from core.classical.ssr import single_scale_retinex
from core.classical.msr import multi_scale_retinex
from core.classical.msrcp import multi_scale_retinex_chromaticity_preservation
from core.classical.agc import adaptive_gamma_correction
from core.classical.clahe import clahe_enhancement
from core.adaptive.hybrid_retinex import hybrid_adaptive_retinex
from core.evaluation.full_reference import calculate_full_reference_metrics
from core.evaluation.no_reference import calculate_no_reference_metrics


def run_comprehensive_study():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    samples_dir = os.path.join(project_root, "data", "samples")
    output_dir = os.path.join(project_root, "data", "outputs")
    os.makedirs(output_dir, exist_ok=True)

    scenes = [
        ("Scene 1: Color Chart", os.path.join(samples_dir, "sample1_colorchart_low.png"), os.path.join(samples_dir, "sample1_colorchart_normal.png")),
        ("Scene 2: Night Campus", os.path.join(samples_dir, "sample2_night_low.png"), os.path.join(samples_dir, "sample2_night_normal.png")),
        ("Scene 3: Indoor Lab", os.path.join(samples_dir, "sample3_indoor_low.png"), os.path.join(samples_dir, "sample3_indoor_normal.png")),
        ("Scene 4: Urban Street", os.path.join(samples_dir, "sample4_street_low.png"), os.path.join(samples_dir, "sample4_street_normal.png")),
        ("Scene 5: Portrait", os.path.join(samples_dir, "sample5_portrait_low.png"), os.path.join(samples_dir, "sample5_portrait_normal.png")),
    ]

    algorithms = {
        "SSR": lambda img: single_scale_retinex(img, sigma=80.0)[:3],
        "MSR": lambda img: multi_scale_retinex(img, sigmas=[15.0, 80.0, 250.0])[:3],
        "MSRCP": lambda img: multi_scale_retinex_chromaticity_preservation(img)[:3],
        "AGCWD": lambda img: adaptive_gamma_correction(img)[:3],
        "CLAHE": lambda img: clahe_enhancement(img, clip_limit=2.5)[:3],
        "Proposed Hybrid Retinex": lambda img: hybrid_adaptive_retinex(img)[:3]
    }

    all_records = []
    print("=" * 85)
    print("STARTING COMPREHENSIVE EXPERIMENTAL STUDY (6 ALGORITHMS x 5 SCENES)")
    print("=" * 85)

    for scene_idx, (scene_name, low_path, norm_path) in enumerate(scenes):
        print(f"\nProcessing {scene_name}...")
        img_low = load_image(low_path)
        img_norm = load_image(norm_path)

        for algo_name, algo_fn in algorithms.items():
            t_start = time.perf_counter()
            enhanced, refl, illum = algo_fn(img_low)
            latency_ms = (time.perf_counter() - t_start) * 1000.0

            # Compute objective metrics
            fr = calculate_full_reference_metrics(enhanced, img_norm)
            nr = calculate_no_reference_metrics(img_low, enhanced)

            # Save enhanced output image
            clean_algo = algo_name.replace(" ", "_").replace("(", "").replace(")", "").lower()
            save_name = f"scene_{scene_idx+1}_{clean_algo}.png"
            save_path = os.path.join(output_dir, save_name)
            save_image(save_path, enhanced)

            record = {
                "Scene": scene_name,
                "Algorithm": algo_name,
                "Input_Brightness": round(float(np.mean(img_low)), 4),
                "Enhanced_Brightness": nr["Mean Brightness"],
                "Brightness_Gain": nr["Brightness Gain"],
                "RMS_Contrast": nr["RMS Contrast"],
                "Entropy": nr["Enhanced Entropy (bits)"],
                "Entropy_Gain": nr["Entropy Gain (bits)"],
                "PSNR_dB": fr["PSNR (dB)"],
                "SSIM": fr["SSIM"],
                "LOE": nr["Lightness Order Error (LOE)"],
                "Naturalness_Index": nr["Naturalness Score (NIQE-proxy)"],
                "Average_Gradient": nr["Average Gradient (Sharpness)"],
                "Latency_ms": round(latency_ms, 2),
                "Output_Image": save_name
            }
            all_records.append(record)
            print(f"  [{algo_name:<23}] PSNR: {record['PSNR_dB']:>5.2f} dB | SSIM: {record['SSIM']:>6.4f} | "
                  f"Entropy: {record['Entropy']:>4.2f} (+{record['Entropy_Gain']:>4.2f}) | LOE: {record['LOE']:>4.2f} | Time: {record['Latency_ms']:>5.1f} ms")

    # 1. Save results to CSV (both in data/outputs and project root)
    csv_paths = [
        os.path.join(output_dir, "benchmark_results.csv"),
        os.path.join(project_root, "benchmark_results.csv")
    ]
    fieldnames = list(all_records[0].keys())
    for cp in csv_paths:
        with open(cp, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_records)
        print(f"[SAVED] Results exported to CSV: {cp}")

    # 2. Save results to JSON
    json_path = os.path.join(output_dir, "benchmark_results.json")
    with open(json_path, "w") as f:
        json.dump(all_records, f, indent=4)
    print(f"[SAVED] Results exported to JSON: {json_path}")

    # 3. Generate Comparative Summary Tables by Algorithm (Mean across 5 scenes)
    print("\n" + "=" * 85)
    print("MEAN PERFORMANCE SUMMARY ACROSS ALL 5 TEST SCENES:")
    print("=" * 85)
    algo_means = {}
    for a in algorithms.keys():
        a_recs = [r for r in all_records if r["Algorithm"] == a]
        algo_means[a] = {
            "PSNR_dB": np.mean([r["PSNR_dB"] for r in a_recs]),
            "SSIM": np.mean([r["SSIM"] for r in a_recs]),
            "Entropy": np.mean([r["Entropy"] for r in a_recs]),
            "Entropy_Gain": np.mean([r["Entropy_Gain"] for r in a_recs]),
            "RMS_Contrast": np.mean([r["RMS_Contrast"] for r in a_recs]),
            "LOE": np.mean([r["LOE"] for r in a_recs]),
            "Naturalness": np.mean([r["Naturalness_Index"] for r in a_recs]),
            "Latency_ms": np.mean([r["Latency_ms"] for r in a_recs]),
        }
        m = algo_means[a]
        print(f"{a:<25} | PSNR: {m['PSNR_dB']:.2f} dB | SSIM: {m['SSIM']:.4f} | Entropy: {m['Entropy']:.2f} | "
              f"Contrast: {m['RMS_Contrast']:.3f} | LOE: {m['LOE']:.2f} | NIQE: {m['Naturalness']:.2f} | Time: {m['Latency_ms']:.1f} ms")

    # 4. Generate Visual Comparison Charts
    generate_comparison_plots(all_records, algo_means, output_dir, project_root)
    print("=" * 85)
    print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("=" * 85)
    return all_records, algo_means


def generate_comparison_plots(all_records, algo_means, output_dir, project_root):
    """Generate high-resolution academic figures for paper and presentation."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    
    # -------------------------------------------------------------------------
    # Chart 1: Mean PSNR and SSIM Comparison Bar Chart
    # -------------------------------------------------------------------------
    algos = list(algo_means.keys())
    psnr_vals = [algo_means[a]["PSNR_dB"] for a in algos]
    ssim_vals = [algo_means[a]["SSIM"] for a in algos]

    x = np.arange(len(algos))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(10, 5), dpi=300)
    color1 = "#2563EB" # Royal Blue
    color2 = "#10B981" # Emerald Green

    rects1 = ax1.bar(x - width/2, psnr_vals, width, label="PSNR (dB)", color=color1, edgecolor="black", linewidth=0.5)
    ax1.set_ylabel("Peak Signal-to-Noise Ratio (dB)", color=color1, fontsize=12, fontweight="bold")
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_ylim(0, max(psnr_vals) * 1.25)

    ax2 = ax1.twinx()
    rects2 = ax2.bar(x + width/2, ssim_vals, width, label="SSIM", color=color2, edgecolor="black", linewidth=0.5)
    ax2.set_ylabel("Structural Similarity Index (SSIM)", color=color2, fontsize=12, fontweight="bold")
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.set_ylim(0, 1.0)

    ax1.set_xticks(x)
    ax1.set_xticklabels([a.replace("Proposed ", "") for a in algos], rotation=15, ha="right", fontsize=10, fontweight="bold")
    ax1.set_title("Quantitative Evaluation: Mean PSNR & SSIM Across 5 Benchmark Scenes", fontsize=13, fontweight="bold", pad=15)
    
    # Annotate bars
    for r in rects1:
        h = r.get_height()
        ax1.annotate(f"{h:.1f}", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold", color=color1)
    for r in rects2:
        h = r.get_height()
        ax2.annotate(f"{h:.3f}", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold", color=color2)

    fig.tight_layout()
    chart1_path = os.path.join(output_dir, "comparison_psnr_ssim.png")
    fig.savefig(chart1_path, dpi=300)
    fig.savefig(os.path.join(project_root, "comparison_psnr_ssim.png"), dpi=300)
    plt.close(fig)
    print(f"[CHART] Saved: {chart1_path}")

    # -------------------------------------------------------------------------
    # Chart 2: Entropy and Contrast Enhancement Comparison
    # -------------------------------------------------------------------------
    entropy_vals = [algo_means[a]["Entropy"] for a in algos]
    contrast_vals = [algo_means[a]["RMS_Contrast"] for a in algos]

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    colors = ["#64748B", "#475569", "#0284C7", "#D97706", "#7C3AED", "#2563EB"]
    bars = ax.bar(algos, entropy_vals, color=colors, edgecolor="black", linewidth=0.5, width=0.55)
    ax.set_ylabel("Shannon Information Entropy (bits)", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 8.5)
    ax.set_xticklabels([a.replace("Proposed ", "") for a in algos], rotation=15, ha="right", fontsize=10, fontweight="bold")
    ax.set_title("Dynamic Range Analysis: Mean Information Entropy (bits)", fontsize=13, fontweight="bold", pad=15)

    for b in bars:
        h = b.get_height()
        ax.annotate(f"{h:.2f} bits", xy=(b.get_x() + b.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    fig.tight_layout()
    chart2_path = os.path.join(output_dir, "comparison_entropy.png")
    fig.savefig(chart2_path, dpi=300)
    fig.savefig(os.path.join(project_root, "comparison_entropy.png"), dpi=300)
    plt.close(fig)
    print(f"[CHART] Saved: {chart2_path}")


if __name__ == "__main__":
    run_comprehensive_study()
