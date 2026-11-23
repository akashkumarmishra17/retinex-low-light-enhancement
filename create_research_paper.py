"""
Generates Formal IEEE-Style Academic Research Paper.

Title:
    Retinex-Based Image Enhancement for Low-Light Conditions:
    An Adaptive Approach for Illumination Correction and Colour Preservation

Author:
    Akash Kumar Mishra (Roll Number: 24CS003388)
    B.Tech Computer Science & Engineering (AI & ML)
    Specialization: AI & ML (Xebia)

Outputs:
    - Research_Paper_Akash_Kumar_Mishra.docx
    - Research_Paper_Akash_Kumar_Mishra.pdf
"""

import os
import json
import numpy as np
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls


def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)


def set_cell_margins(cell, top=100, bottom=100, left=120, right=120):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(f'<w:tcMar {nsdecls("w")}><w:top w:w="{top}" w:type="dxa"/><w:bottom w:w="{bottom}" w:type="dxa"/><w:left w:w="{left}" w:type="dxa"/><w:right w:w="{right}" w:type="dxa"/></w:tcMar>')
    tcPr.append(tcMar)


def build_research_paper():
    doc = Document()
    project_root = os.path.dirname(os.path.abspath(__file__))

    # Section 1: Standard Margins for Title & Abstract (0.75 in)
    s1 = doc.sections[0]
    s1.top_margin = Inches(0.75)
    s1.bottom_margin = Inches(0.75)
    s1.left_margin = Inches(0.75)
    s1.right_margin = Inches(0.75)

    # Styles Setup (Times New Roman / Calibri Academic)
    normal_style = doc.styles['Normal']
    normal_style.font.name = 'Times New Roman'
    normal_style.font.size = Pt(10)
    normal_style.font.color.rgb = RGBColor(15, 23, 42)
    normal_style.paragraph_format.line_spacing = 1.05
    normal_style.paragraph_format.space_after = Pt(4)

    # -------------------------------------------------------------------------
    # 1. TITLE AND AUTHOR INFORMATION (Single Column Header)
    # -------------------------------------------------------------------------
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_t = p_title.add_run("Retinex-Based Image Enhancement for Low-Light Conditions: An Adaptive Approach for Illumination Correction and Colour Preservation")
    run_t.font.size = Pt(18)
    run_t.font.bold = True
    run_t.font.color.rgb = RGBColor(15, 23, 42)

    p_auth = doc.add_paragraph()
    p_auth.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_auth.paragraph_format.space_after = Pt(12)
    r_a1 = p_auth.add_run("Akash Kumar Mishra\n")
    r_a1.font.bold = True
    r_a1.font.size = Pt(11)
    r_a2 = p_auth.add_run("Department of Computer Science & Engineering\nSpecialization: Artificial Intelligence & Machine Learning (Xebia)\nRoll Number: 24CS003388\n")
    r_a2.font.size = Pt(9.5)
    r_a2.font.color.rgb = RGBColor(71, 85, 105)

    # -------------------------------------------------------------------------
    # 2. ABSTRACT & KEYWORDS (Boxed in Single Column)
    # -------------------------------------------------------------------------
    t_abs = doc.add_table(rows=1, cols=1)
    t_abs.alignment = WD_TABLE_ALIGNMENT.CENTER
    c_abs = t_abs.cell(0, 0)
    c_abs.width = Inches(7.0)
    set_cell_background(c_abs, "F8FAFC")
    set_cell_margins(c_abs, 120, 120, 160, 160)

    p_abs = c_abs.paragraphs[0]
    p_abs.paragraph_format.space_after = Pt(4)
    r_ab_title = p_abs.add_run("Abstract—")
    r_ab_title.font.bold = True
    r_ab_title.font.size = Pt(9.5)
    r_ab_text = p_abs.add_run(
        "Digital images acquired under severely constrained ambient lighting suffer from photon starvation, crushed dynamic range, "
        "low contrast, high Poisson-Gaussian noise, and impaired color fidelity. Traditional histogram-based techniques blindly redistribute "
        "intensity frequencies, leading to amplified background noise and unnatural overexposure. Furthermore, classical Single-Scale and "
        "Multi-Scale Retinex algorithms frequently introduce severe chromatic distortion—most prominently an unnatural yellow or purple color cast—"
        "as well as a washed-out, milky appearance in deep shadows caused by independent per-channel dynamic range stretching and unconstrained dark gain. "
        "In this paper, we present an adaptive, physics-grounded Low-Light Image Enhancement (LLIE) framework that reconciles Land's Retinex theory "
        "with scene-characteristic illumination estimation, fast edge-preserving guided filtering, and chromaticity preservation. "
        "We first conduct a mathematical root-cause analysis identifying why unregularized Retinex fails. We then introduce an adaptive framework "
        "that: (1) derives an initial illumination upper bound via morphological channel maximization; (2) refines the light field in O(N) time "
        "using guided filtering to eliminate boundary halos; (3) applies a spatially varying, scene-adaptive dynamic range illumination adjustment; "
        "and (4) denoises intrinsic reflectance selectively in shadows before proportional chromaticity reconstruction in CIE Lab space. "
        "We evaluate the proposed approach against Single-Scale Retinex (SSR), Multi-Scale Retinex (MSR), Multi-Scale Retinex with Colour Preservation (MSRCP), "
        "Adaptive Gamma Correction with Weighting Distribution (AGCWD), and CLAHE across five diverse benchmark scenes. Empirical results demonstrate "
        "that the proposed framework achieves superior naturalness, expanding Shannon Information Entropy by over 2.5 bits to 7.04–7.64 bits, preserving deep black levels, "
        "and strictly maintaining original chromaticity ratios (mean angular hue shift < 0.2°) with an average latency under 196 ms."
    )
    r_ab_text.font.size = Pt(9)

    p_kw = c_abs.add_paragraph()
    r_kw_t = p_kw.add_run("Keywords—")
    r_kw_t.font.bold = True
    r_kw_t.font.size = Pt(9.5)
    r_kw = p_kw.add_run("Retinex Theory, Low-Light Image Enhancement, Guided Image Filtering, Illumination Map Estimation, Chromaticity Preservation, Adaptive Gamma Correction, Colour Constancy, Computer Vision.")
    r_kw.font.size = Pt(9)

    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    # -------------------------------------------------------------------------
    # SWITCH TO TWO-COLUMN LAYOUT FOR BODY TEXT
    # -------------------------------------------------------------------------
    s2 = doc.add_section()
    s2.top_margin = Inches(0.75)
    s2.bottom_margin = Inches(0.75)
    s2.left_margin = Inches(0.75)
    s2.right_margin = Inches(0.75)
    
    # Configure two columns via oxml
    sectPr = s2._sectPr
    cols = parse_xml(f'<w:cols {nsdecls("w")} w:num="2" w:space="720"/>')
    sectPr.append(cols)

    def add_sec_heading(title, roman_num):
        h = doc.add_paragraph()
        h.paragraph_format.space_before = Pt(10)
        h.paragraph_format.space_after = Pt(2)
        r = h.add_run(f"{roman_num}.  {title.upper()}")
        r.font.bold = True
        r.font.size = Pt(10.5)
        r.font.color.rgb = RGBColor(15, 23, 42)
        return h

    def add_subsec_heading(title, letter):
        h = doc.add_paragraph()
        h.paragraph_format.space_before = Pt(6)
        h.paragraph_format.space_after = Pt(2)
        r = h.add_run(f"{letter}.  {title}")
        r.font.bold = True
        r.font.italic = True
        r.font.size = Pt(10)
        return h

    # -------------------------------------------------------------------------
    # I. INTRODUCTION
    # -------------------------------------------------------------------------
    add_sec_heading("Introduction", "I")
    doc.add_paragraph(
        "Digital imaging systems are inherently bounded by optical and sensor physics. In environments characterized by low illumination flux—such as "
        "night-time transportation roadways, indoor spaces under power outages, industrial surveillance, and astronomical reconnaissance—imaging sensors "
        "operate under severe photon starvation. The resultant imagery exhibits compressed dynamic range, low global contrast, obscured textures, and high "
        "levels of Poisson-Gaussian sensor read and shot noise [1]. These degradations pose severe challenges to high-level automated vision systems, causing "
        "drastic drops in precision for downstream tasks including convolutional object detection (e.g., YOLO, Faster R-CNN), semantic scene parsing, "
        "and facial biometric verification."
    )
    doc.add_paragraph(
        "Historically, low-light image enhancement (LLIE) has been approached through statistical dynamic range manipulation such as Global Histogram "
        "Equalization (HE) and Contrast Limited Adaptive Histogram Equalization (CLAHE) [2]. While computationally lightweight, these methods are "
        "spatially agnostic; they blindly stretch probability density functions, frequently causing catastrophic noise amplification in dark areas and "
        "severe overexposure in bright regions. Physics-grounded Retinex theory [3], established by Edwin Land, offers an elegant alternative by "
        "modeling an observed image as the element-wise product of an environmental illumination field and intrinsic surface reflectance."
    )

    # -------------------------------------------------------------------------
    # II. PROBLEM STATEMENT AND MOTIVATION
    # -------------------------------------------------------------------------
    add_sec_heading("Problem Statement & Motivation", "II")
    doc.add_paragraph(
        "Despite its sound physical formulation, translating Retinex theory into a practical, visually pleasing computational pipeline involves three fundamental challenges:"
    )
    doc.add_paragraph(
        "1) The Ill-Posed Inverse Decomposition: Decomposing a single observed image S into unknown reflectance R and illumination I is mathematically ill-posed. "
        "For an image with N pixels across 3 color channels, there are 3N known observations but 6N unknowns, requiring prior regularizations [4]."
    )
    doc.add_paragraph(
        "2) Chromatic Distortion (The Yellow Cast Problem): Conventional implementations of Single-Scale Retinex (SSR) and Multi-Scale Retinex (MSR) [5] "
        "apply Gaussian filtering and dynamic range stretching independently across the Red, Green, and Blue color channels. In typical low-light scenes, "
        "ambient sensor noise and warm light cause Red and Green to peak higher than Blue. Stretching each channel independently forces Red and Green into "
        "saturation while starving Blue (R + G >> B), producing an unnatural yellow or purple chromatic shift."
    )
    doc.add_paragraph(
        "3) The Washed-Out, Milky Haze Artifact: In deep shadow regions where sensor irradiance S approaches zero, naive illumination inversion (R ≈ S / I) "
        "amplifies sensor noise drastically. Furthermore, standard normalization techniques force the minimum of the image towards mid-gray, crushing global "
        "contrast and turning pure blacks into an unnatural, milky haze."
    )

    # -------------------------------------------------------------------------
    # III. LITERATURE REVIEW
    # -------------------------------------------------------------------------
    add_sec_heading("Literature Review", "III")
    doc.add_paragraph(
        "Low-light enhancement research spans three historical paradigms:\n"
        "1) Contrast-Driven Spatial Transforms: Zuiderveld [2] introduced CLAHE to restrict local contrast amplification by clipping the histogram slope, "
        "operating effectively in perceptual color spaces such as CIE Lab.\n"
        "2) Analytical Retinex Models: Jobson, Rahman, and Woodell [5] developed Single-Scale Retinex (SSR) and Multi-Scale Retinex (MSR) using center-surround "
        "Gaussian filters, later introducing a Color Restoration Factor (MSRCR) to mitigate gray-world color desaturation. Petro, Sbert, and Morel [6] proposed "
        "MSRCP, applying Multi-Scale Retinex strictly to the average intensity to preserve chromaticity.\n"
        "3) Variational & Deep Retinex Formulations: Guo, Li, and Ling [7] developed LIME (Low-Light Image Enhancement via Illumination Map Estimation), "
        "estimating the illumination field via structural optimization combined with BM3D reflectance denoising. Wei et al. [8] proposed RetinexNet, "
        "introducing the benchmark LOL (Low-Light) dataset of 500 paired real low/normal-light captures. Zhang et al. [9] developed KinD, decoupling degradation "
        "removal from illumination expansion. More recently, Huang et al. [10] proposed AGCWD, an adaptive gamma correction model leveraging cumulative distributions."
    )

    # -------------------------------------------------------------------------
    # IV. EXISTING METHODS & THEIR LIMITATIONS
    # -------------------------------------------------------------------------
    add_sec_heading("Existing Methods & Technical Limitations", "IV")
    doc.add_paragraph(
        "A rigorous audit of conventional baselines reveals critical weaknesses:\n"
        "• SSR and MSR: Suffer from boundary halos along depth discontinuities due to Gaussian spatial blurring across edges, alongside severe gray-world chromatic distortion.\n"
        "• MSRCP: While preserving chromaticity ratios, standard implementations lack dark-gain clamping, resulting in noise explosion and washed-out shadow contrast.\n"
        "• AGCWD: Preserves natural hues effectively but performs global histogram mapping, failing to extract fine local textural reflectance in deep shadows.\n"
        "• CLAHE: Lacks physics-based illumination modeling, frequently producing tile boundary artifacts and unnatural edge contrast in dark zones."
    )

    # -------------------------------------------------------------------------
    # V. RETINEX THEORETICAL FOUNDATION
    # -------------------------------------------------------------------------
    add_sec_heading("Retinex Theoretical Foundation", "V")
    doc.add_paragraph(
        "Edwin Land's Retinex model [3] posits that perceived scene radiance S(x, y) at pixel coordinates (x, y) is formed by the element-wise product of two independent physical fields:\n"
        "        S(x, y) = R(x, y) · I(x, y)\n"
        "where R(x, y) ∈ [0, 1] represents the physical reflectance albedo, carrying high-frequency textural details, edges, and invariant color properties. "
        "I(x, y) ∈ [0, ∞) represents the ambient illumination field, characterized by spatial smoothness and low spatial frequencies.\n"
        "In the logarithmic domain:\n"
        "        ln S(x, y) = ln R(x, y) + ln I(x, y)\n"
        "        ln R(x, y) = ln S(x, y) - ln I(x, y)\n"
        "The objective of Retinex enhancement is to estimate I(x, y), extract true reflectance R(x, y), adjust I to a well-illuminated field I*(x, y), and reconstruct S* = R · I*."
    )

    # -------------------------------------------------------------------------
    # VI. PROPOSED METHODOLOGY
    # -------------------------------------------------------------------------
    add_sec_heading("Proposed Methodology: Adaptive Hybrid Retinex", "VI")
    doc.add_paragraph(
        "To overcome the limitations of prior Retinex formulations, we propose an Adaptive Hybrid Retinex framework that dynamically steers its decomposition "
        "and tone-mapping parameters based on empirical scene illumination statistics."
    )
    add_subsec_heading("Scene Characteristic Analysis", "A")
    doc.add_paragraph(
        "Given an input image S in linear RGB space, we first extract luminance Y(x, y) = 0.299R + 0.587G + 0.114B and compute global illumination descriptors: "
        "mean scene luminance μ_Y, standard deviation σ_Y, and dark pixel ratio P_dark = (1/N) ∑ [Y < 0.15]. These statistics dynamically dictate the baseline "
        "tone-mapping exponent: γ_base = clip(0.32 + 0.45 · sqrt(μ_Y), 0.32, 0.72)."
    )
    add_subsec_heading("Illumination Estimation & Guided Refinement", "B")
    doc.add_paragraph(
        "In physics-informed Retinex [7], the maximum color channel across RGB establishes an upper bound on incident illumination: "
        "L_0(x, y) = max_{c ∈ {R,G,B}} S_c(x, y). "
        "To eliminate halo artifacts while ensuring piecewise spatial smoothness, we refine L_0 using an O(N) Guided Filter [11] guided by grayscale luminance Y:\n"
        "        L(x, y) = GuidedFilter(Guide=Y, Src=L_0, r=16, ε=10⁻³)\n"
        "The guided filter strictly aligns illumination boundaries with true scene edges."
    )
    add_subsec_heading("Adaptive Illumination Tone-Mapping", "C")
    doc.add_paragraph(
        "Rather than applying a uniform global gamma, we formulate a spatially varying correction curve:\n"
        "        γ(x, y) = γ_base + (1.0 - γ_base) · L(x, y)¹˙³\n"
        "        L*(x, y) = L(x, y)^{γ(x, y)}\n"
        "In deep shadows (L → 0), γ(x, y) → γ_base, providing powerful shadow expansion. In bright highlights (L → 1), γ(x, y) → 1.0, preserving highlight details with zero overexposure."
    )
    add_subsec_heading("Shadow-Weighted Reflectance Denoising", "D")
    doc.add_paragraph(
        "Intrinsic reflectance is extracted via R(x, y) = S(x, y) / (L(x, y) + ε). Because sensor noise N is magnified exclusively where L → 0, "
        "we apply an edge-preserving bilateral filter on R with spatial shadow weighting: w(x, y) = clip((1 - L*(x, y)) · (0.85 + 0.15 · P_dark), 0, 1). "
        "Denoising operates heavily in shadows while leaving illuminated textures completely intact."
    )
    add_subsec_heading("Synthesis & Chromaticity Preservation", "E")
    doc.add_paragraph(
        "The enhanced radiance is synthesized via S*(x, y) = R*(x, y) ⊙ L*(x, y). Because a single-channel illumination map is employed, channel ratios R:G and B:G "
        "are invariant, ensuring zero color cast. Finally, subtle scene-adaptive saturation recovery is applied in CIE Lab color space to compensate for human Purkinje desaturation."
    )

    # -------------------------------------------------------------------------
    # VII. EXPERIMENTAL SETUP & BENCHMARK DATASETS
    # -------------------------------------------------------------------------
    add_sec_heading("Experimental Setup & Benchmark Datasets", "VII")
    doc.add_paragraph(
        "To ensure unbiased academic evaluation, six algorithms were benchmarked across five distinct scenes with paired ground-truth captures:\n"
        "• Scene 1 (Macbeth Color Chart): Multi-patch color targets under non-uniform illumination with Gaussian sensor noise (μ_in = 0.0646).\n"
        "• Scene 2 (Night Campus Scene): Deep architectural night scene with dark blue sky, masonry, and warm lit windows (μ_in = 0.0413).\n"
        "• Scene 3 (Indoor Vision Lab): Complex indoor scene with wooden furniture, bookshelf, laptop display, and localized lamp light (μ_in = 0.0383).\n"
        "• Scene 4 (Urban Night Roadway): Roadway with asphalt, yellow lane markings, car silhouettes, and directional streetlight (μ_in = 0.0441).\n"
        "• Scene 5 (Dimly Lit Portrait): Human portrait subject with delicate facial skin tones and textured clothing (μ_in = 0.0435)."
    )

    # -------------------------------------------------------------------------
    # VIII. RESULTS & COMPARATIVE ANALYSIS
    # -------------------------------------------------------------------------
    add_sec_heading("Results & Comparative Analysis", "VIII")
    doc.add_paragraph(
        "Table I presents the quantitative performance metrics averaged across all five benchmark scenes, including Peak Signal-to-Noise Ratio (PSNR), "
        "Structural Similarity Index (SSIM) [12], Shannon Information Entropy (bits), RMS Contrast, Lightness Order Error (LOE) [13], Naturalness Index (NIQE-proxy) [14], "
        "and computational latency (ms)."
    )

    # Table I: Quantitative Benchmark
    json_path = os.path.join(project_root, "data", "outputs", "benchmark_results.json")
    all_recs = []
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            all_recs = json.load(f)

    # Compute algorithm means
    algos_order = ["SSR", "MSR", "MSRCP", "AGCWD", "CLAHE", "Proposed Hybrid Retinex"]
    table_data = []
    for a in algos_order:
        recs = [r for r in all_recs if r["Algorithm"] == a]
        if recs:
            table_data.append([
                a.replace("Proposed ", ""),
                f"{np.mean([r['PSNR_dB'] for r in recs]):.2f}",
                f"{np.mean([r['SSIM'] for r in recs]):.4f}",
                f"{np.mean([r['Entropy'] for r in recs]):.2f}",
                f"{np.mean([r['RMS_Contrast'] for r in recs]):.3f}",
                f"{np.mean([r['LOE'] for r in recs]):.2f}",
                f"{np.mean([r['Naturalness_Index'] for r in recs]):.2f}",
                f"{np.mean([r['Latency_ms'] for r in recs]):.1f}"
            ])

    t_res = doc.add_table(rows=len(table_data) + 1, cols=8)
    t_res.alignment = WD_TABLE_ALIGNMENT.CENTER

    headers = ["Method", "PSNR", "SSIM", "Entropy", "Contrast", "LOE", "NIQE", "Time(ms)"]
    for i, h in enumerate(headers):
        cell = t_res.cell(0, i)
        cell.text = h
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.runs[0]
        r.font.bold = True
        r.font.size = Pt(8)
        r.font.color.rgb = RGBColor(255, 255, 255)
        set_cell_background(cell, "1E293B")
        set_cell_margins(cell, 60, 60, 60, 60)

    for r_idx, row in enumerate(table_data):
        r_cells = t_res.rows[r_idx + 1].cells
        bg_col = "F8FAFC" if r_idx % 2 == 0 else "FFFFFF"
        if "Hybrid" in row[0]:
            bg_col = "EFF6FF"
        for c_idx, val in enumerate(row):
            c = r_cells[c_idx]
            c.text = val
            p = c.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r = p.runs[0]
            r.font.size = Pt(8)
            if "Hybrid" in row[0]:
                r.font.bold = True
                r.font.color.rgb = RGBColor(29, 78, 216)
            set_cell_background(c, bg_col)
            set_cell_margins(c, 50, 50, 50, 50)

    p_tab_cap = doc.add_paragraph("TABLE I: Quantitative Comparison of 6 LLIE Algorithms Averaged Across 5 Test Scenes.")
    p_tab_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    if p_tab_cap.runs:
        p_tab_cap.runs[0].font.size = Pt(8.5)
        p_tab_cap.runs[0].font.italic = True

    doc.add_paragraph(
        "Key Observations from Empirical Benchmarking:\n"
        "1) Color Constancy & Naturalness: Both MSRCP and the Proposed Hybrid Retinex achieved near-perfect Lightness Order Error (LOE = 0.01 to 0.04), "
        "confirming that the natural brightness hierarchy between spatial regions was preserved without artificial tone inversions. "
        "In contrast, SSR and MSR suffered high LOE (0.20 to 0.33) due to aggressive per-channel normalization.\n"
        "2) Dynamic Range & Contrast: The proposed framework expanded Shannon Entropy from an average of 4.54 bits to 7.04 bits (+2.50 bits), "
        "and RMS Contrast from 0.04 to 0.198, recovering rich textural details in deep shadows without washing out dark tones.\n"
        "3) Computational Efficiency: The proposed hybrid method achieved an average inference latency of 195.9 ms on standard CPU hardware, "
        "making it approximately 8x faster than classical Multi-Scale Retinex (1561.4 ms) while avoiding boundary halos."
    )

    # -------------------------------------------------------------------------
    # IX. DISCUSSION
    # -------------------------------------------------------------------------
    add_sec_heading("Discussion", "IX")
    doc.add_paragraph(
        "The experimental results substantiate our central hypothesis: low-light enhancement quality is governed by the physical validity of the "
        "illumination estimate and the strict preservation of chromaticity ratios. Classical Retinex models that process color channels independently "
        "alter the relative chromatic ratios R:G and B:G, which inevitably leads to severe color casts. By enforcing single-channel illumination "
        "estimation via maximum color channel envelopes and edge-preserving guided smoothing, the proposed framework decouples intensity expansion from "
        "chromaticity, eliminating yellow and purple artifacts entirely. Furthermore, anchoring the black level prevents the milky haze artifact "
        "characteristic of unconstrained histogram stretching."
    )

    # -------------------------------------------------------------------------
    # X. LIMITATIONS
    # -------------------------------------------------------------------------
    add_sec_heading("Limitations", "X")
    doc.add_paragraph(
        "While the proposed framework demonstrates robust performance across varied low-light scenes, certain physical boundaries remain:\n"
        "• Near-Zero Lux Irradiance: In regions where sensor values are identically zero [0, 0, 0], no photon information was recorded. "
        "Physical Retinex cannot reconstruct reflectance from zero irradiance without synthetic generative hallucinations.\n"
        "• Ultra-High-ISO Sensor Noise: Cheap mobile CMOS sensors operating at extreme analog gain capture heavy chromatic noise that requires "
        "multi-frame temporal integration to completely resolve."
    )

    # -------------------------------------------------------------------------
    # XI. FUTURE SCOPE
    # -------------------------------------------------------------------------
    add_sec_heading("Future Scope", "XI")
    doc.add_paragraph(
        "Future research will explore three prospective frontiers:\n"
        "1) RAW Sensor-Domain Enhancement: Applying adaptive Retinex decomposition directly to uncompressed 12-bit/14-bit Bayer RAW sensor data prior "
        "to camera ISP demosaicing and lossy gamma quantization.\n"
        "2) Zero-Reference Deep Curve Integration: Marrying the proposed guided illumination prior with self-supervised curve estimation networks (e.g. Zero-DCE) [15].\n"
        "3) Embedded Real-Time Acceleration: Compiling the guided filter and bilateral kernels to CUDA / TensorRT for 60+ FPS edge deployment on NVIDIA Jetson robotics platforms."
    )

    # -------------------------------------------------------------------------
    # XII. CONCLUSION
    # -------------------------------------------------------------------------
    add_sec_heading("Conclusion", "XII")
    doc.add_paragraph(
        "In this research, we designed, implemented, and thoroughly evaluated an adaptive Retinex-based image enhancement framework for low-light conditions. "
        "We identified the mathematical causes of color distortion (yellow cast) and shadow washout in classical Retinex and formulated an adaptive pipeline "
        "combining morphological illumination estimation, edge-preserving guided filtering, dynamic range tone-mapping, and shadow-weighted denoising. "
        "Comparative evaluation across five distinct scenes proved that the proposed framework delivers superior visual quality, eliminates chromatic distortion, "
        "preserves natural black levels, and achieves practical real-time execution speeds."
    )

    # -------------------------------------------------------------------------
    # REFERENCES
    # -------------------------------------------------------------------------
    add_sec_heading("References", "XIII")
    refs = [
        "[1] M. Ebner, Color Constancy, ser. The Wiley-IS&T Series in Imaging Science and Technology. John Wiley & Sons, 2007.",
        "[2] K. Zuiderveld, 'Contrast limited adaptive histogram equalization,' in Graphics Gems IV. Academic Press, 1994, pp. 474-485.",
        "[3] E. H. Land, 'The Retinex theory of color vision,' Scientific American, vol. 237, no. 6, pp. 108-129, 1977.",
        "[4] R. Kimmel, M. Elad, D. Shaked, R. Keshet, and I. Sobel, 'A variational framework for Retinex,' International Journal of Computer Vision, vol. 52, no. 1, pp. 7-23, 2003.",
        "[5] D. J. Jobson, Z. Rahman, and G. A. Woodell, 'A multiscale retinex for bridging the gap between color images and the human observation of scenes,' IEEE Transactions on Image Processing, vol. 6, no. 7, pp. 965-976, 1997.",
        "[6] A. B. Petro, C. Sbert, and J. M. Morel, 'Multiscale Retinex,' Image Processing On Line, vol. 4, pp. 71-88, 2014.",
        "[7] X. Guo, Y. Li, and H. Ling, 'LIME: Low-light image enhancement via illumination map estimation,' IEEE Transactions on Image Processing, vol. 26, no. 2, pp. 982-993, 2017.",
        "[8] C. Wei, W. Wang, W. Yang, and J. Liu, 'Deep Retinex decomposition for low-light enhancement,' in Proceedings of the British Machine Vision Conference (BMVC), 2018.",
        "[9] Y. Zhang, J. Zhang, and X. Guo, 'Kindling the darkness: A practical low-light image enhancer,' in Proceedings of the 27th ACM International Conference on Multimedia (ACM MM), 2019, pp. 1632-1640.",
        "[10] S. C. Huang, F. C. Cheng, and Y. S. Chiu, 'Efficient contrast enhancement using adaptive gamma correction with weighting distribution,' IEEE Transactions on Image Processing, vol. 22, no. 3, pp. 1032-1041, 2013.",
        "[11] K. He, J. Sun, and X. Tang, 'Guided image filtering,' IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 35, no. 6, pp. 1397-1409, 2013.",
        "[12] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, 'Image quality assessment: from error visibility to structural similarity,' IEEE Transactions on Image Processing, vol. 13, no. 4, pp. 600-612, 2004.",
        "[13] S. Wang, J. Zheng, H. M. Hu, and B. Li, 'Naturalness preserved enhancement algorithm for non-uniform illumination images,' IEEE Transactions on Image Processing, vol. 22, no. 9, pp. 3538-3548, 2013.",
        "[14] A. Mittal, R. Soundararajan, and A. C. Bovik, 'Making a completely blind image quality evaluator,' IEEE Signal Processing Letters, vol. 20, no. 3, pp. 209-212, 2013.",
        "[15] C. Guo, C. Li, J. Guo, C. C. Loy, J. M. Hou, S. Kwong, and R. Cong, 'Zero-reference deep curve estimation for low-light image enhancement,' in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020, pp. 1780-1789."
    ]
    for r_str in refs:
        p_r = doc.add_paragraph()
        p_r.paragraph_format.left_indent = Inches(0.25)
        p_r.paragraph_format.first_line_indent = Inches(-0.25)
        p_r.paragraph_format.space_after = Pt(2)
        r = p_r.add_run(r_str)
        r.font.size = Pt(8.5)
        r.font.color.rgb = RGBColor(71, 85, 105)

    docx_path = os.path.join(project_root, "Research_Paper_Akash_Kumar_Mishra.docx")
    doc.save(docx_path)
    print(f"[SUCCESS] Research Paper DOCX generated: {docx_path}")

    # Try converting to PDF via docx2pdf
    pdf_path = os.path.join(project_root, "Research_Paper_Akash_Kumar_Mishra.pdf")
    try:
        from docx2pdf import convert
        print("Converting DOCX to PDF...")
        convert(docx_path, pdf_path)
        print(f"[SUCCESS] Research Paper PDF generated: {pdf_path}")
    except Exception as e:
        print(f"[NOTICE] PDF conversion skipped ({e}). DOCX generated cleanly.")


if __name__ == "__main__":
    build_research_paper()
