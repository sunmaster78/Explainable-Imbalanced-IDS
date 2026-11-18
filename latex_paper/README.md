# IEEE Format Paper: Two-Stage Explainable Intrusion Detection System

This directory contains the publication-grade IEEE journal / conference manuscript written in LaTeX, ready for direct compilation on [Overleaf](https://www.overleaf.com) or local LaTeX distributions (`pdflatex`, `latexmk`, TeX Live, MiKTeX).

---

## 1. Directory Structure

```text
latex_paper/
├── main.tex                 # Master IEEE LaTeX manuscript (IEEEtran format)
├── references.bib           # Complete BibTeX bibliography (19 literature papers + foundations)
├── overleaf_project.zip     # Pre-packaged ZIP bundle ready for 1-click Overleaf upload
├── figures/                 # High-resolution 300 DPI PNG & PDF vector figures
│   ├── framework_architecture.png
│   ├── framework_architecture.pdf
│   ├── fig3_per_class_recall_benchmark.png
│   ├── fig3_per_class_recall_benchmark.pdf
│   ├── stage1_threshold_tuning.png
│   ├── wgan_loss_curves.png
│   ├── nslkdd_train_5class_distribution.png
│   └── nslkdd_train_binary_distribution.png
└── README.md                # This user guide
```

---

## 2. Quick Overleaf Setup (1-Click Upload)

1. Open [Overleaf](https://www.overleaf.com) and log in.
2. Click **New Project** $\rightarrow$ **Upload Project**.
3. Select and drag-and-drop the pre-packaged zip file:
   `latex_paper/overleaf_project.zip`
4. Overleaf will automatically unpack all files, set `main.tex` as the root document, and compile the manuscript with all figures and citations.

---

## 3. Local Compilation Instructions

If compiling on your local workstation with TeX Live or MiKTeX:

```bash
cd latex_paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Or using `latexmk`:
```bash
latexmk -pdf main.tex
```

---

## 4. Key Highlights & Methodological Claims in the Paper

1. **Novelty Claim 1 (SHAP Fidelity):** Formulates Spearman rank correlation ($\rho_s = 0.881$), top-10 Jaccard similarity ($J_{10} = 0.750$), and cosine attribution similarity ($S_C = 0.948$) between synthetic and real minority samples, empirically proving that WGAN-GP generates features with higher attribution authenticity than SMOTE or ROS.
2. **Novelty Claim 2 (Stage-Wise SHAP):** Deconstructs model explanations into Stage~1 (normal vs. anomaly filtering) and Stage~2 (inter-attack family differentiation), eliminating the semantic explanation conflation observed in single-stage models.
3. **Novelty Claim 3 (Stage-Aware GAN):** Demonstrates that conditioning a WGAN-GP strictly on attack samples prevents mode contamination from the massive benign traffic class, improving training stability and downstream minority classification.
4. **Empirical Rigor & Zero Data Leakage:**
   - Evaluated across **NSL-KDD**, **UNSW-NB15**, and **CIC-IDS2017**.
   - Incorporates all 19 verified papers from `resorce/LR.docx` into the literature review and comparative analysis.
   - Stage~1 threshold calibration ensures $\ge 99.9\%$ attack recall with sub-millisecond per-sample latency ($0.0071$~ms/flow, $>140,000$ flows/second).
