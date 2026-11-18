# Systematic Literature Review, Taxonomy Matrix, and Novelty Differentiation

> **Document Purpose:** Positioning the Two-Stage Explainable Intrusion Detection System against the 19 contemporary state-of-the-art literature papers for publication in top-tier IEEE/Elsevier journals (e.g., *IEEE Transactions on Information Forensics and Security*, *IEEE Internet of Things Journal*, *Computers & Security*).  
> **Source Repository:** `sunmaster78/Explainable-Imbalanced-IDS`  
> **Companion Word Document:** [`resorce/LR.docx`](file:///e:/BRM003-ML-Project/resorce/LR.docx)  
> **Original Unaltered Draft Backup:** [`resorce/LR_original.docx`](file:///e:/BRM003-ML-Project/resorce/LR_original.docx)

---

## Executive Summary: Differentiation & Publication Positioning

This document establishes the theoretical justification, taxonomic comparison, and empirical differentiation of our proposed **Two-Stage Explainable Intrusion Detection System (IDS)** against the 19 contemporary research papers compiled in the project literature repository.

While prior works have independently explored hierarchical filtering, tree ensemble combinations, generative oversampling (SMOTE/CTGAN/WGAN), or post-hoc XAI (SHAP/LIME), they leave three pivotal scientific voids:
1. **Zero Explanation Validation of Synthetic Data:** None validate whether synthetic minority attack samples preserve the authentic decision manifold of genuine attacks using quantitative feature attribution fidelity metrics.
2. **Semantic Conflation in Monolithic XAI:** None decouple SHAP explanations across a two-stage boundary, causing severe conflation between normal traffic filtering and inter-attack family discrimination.
3. **Mode Contamination in Flat GANs:** Prior studies train GANs across both normal and attack records, allowing the massive benign class (53%–90% of data) to dominate generator attention and trigger mode collapse on rare minority classes (U2R/R2L).

Our framework addresses all three voids with a leak-free architecture, providing empirical proof for three novelty claims.

---

## 1. Systematic Synthesis of the 19 Literature Papers

### 1.1 Group A: Two-Stage and Hierarchical Detection Frameworks
Hierarchical decomposition balances detection granularity against computational latency.
* **Mahmoud et al. (2025) [2] (XI2S-IDS):** Utilizes binary filtering to discard benign packets before routing anomalies to a multi-class classifier. However, their Stage 2 relies on standard unweighted heuristics without generative oversampling, leading to low recall on rare minority classes.
* **Ma et al. (2025) [4]:** Formulates a dynamic feedback cost-sensitive framework combining staged detection with Shapley-value feature selection, yet does not incorporate generative oversampling.
* **Barnard et al. (2022) [1]:** Couples an XGBoost detector with a secondary autoencoder trained on SHAP attributions to identify unseen zero-day attacks, highlighting the value of XAI in staged pipelines.
* **Uddin et al. (2025) [5]:** Implements a 3-level hierarchical IDS for IoV, segmenting binary traffic from DoS and spoofing, though designed specifically for vehicular telemetry rather than enterprise IP flows.
* **Zhang et al. (2023) [14]:** Demonstrates an ultra-fast two-stage IoT-IDS deploying LightGBM for Stage 1 screening (reaching 99.87% precision), confirming the computational efficiency of tree-based filtering.

### 1.2 Group B: Ensemble Machine Learning in NIDS
Tree ensemble architectures combine complementary hypotheses to minimize structural variance and bias across tabular network flow features.
* **Gupta et al. (2025) [9]:** Evaluates multi-class attack classification across NSL-KDD, confirming that an ensemble of LightGBM, XGBoost, and Random Forest consistently outperforms individual tree classifiers in multi-class recall.
* **Muhammad et al. (2025) [6] (xIDS-EnsembleGuard):** Combines tree ensembles with deep neural distillation to achieve 98.8% accuracy on CIC-IDS2017.
* **Alatawi (2025) [8]:** Evaluates ensemble blending with Explainable AI for smart grid infrastructures.
* **Alabdulatif (2025) [12] & Bibers et al. (2025) [13]:** Evaluate stacking frameworks for enterprise intrusion detection, proving that multi-model soft voting provides superior resilience against evasion perturbations.

### 1.3 Group C: Generative Adversarial Networks for Class Imbalance
Class imbalance represents the primary bottleneck in reliable intrusion classification.
* **Yi-fan (2025) [3] (CSAGC-IDS):** Proposes a self-attention conditional GAN to balance minority records in NSL-KDD.
* **Bedi et al. (2021) [15] (I-SiamIDS):** Utilizes Siamese networks to improve minority class separation.
* **Alonso and Pellicer (2026) [18]:** Explores forensic flow synthesis with CTGAN on CICIDS2017, demonstrating that synthetic flows can retain forensic signatures.
* **Zhang et al. (2020) [19] (CWGAN-CSSAE):** Proposes Conditional WGAN with gradient penalty and L2 regularization to balance minority attacks (U2R and R2L) on NSL-KDD and UNSW-NB15, boosting U2R recall to 42.50%.
* *Crucial Limitation:* All existing GAN papers train their generative models "flatly" across the entire dataset (including benign traffic), diluting generator attention and inducing mode collapse.

### 1.4 Group D: Explainable AI (XAI) for Cybersecurity Verification
Post-hoc explainability is essential for operational deployment in Security Operations Centers (SOCs).
* **El-Nashar et al. (2025) [7]:** Integrates SHAP to isolate discriminative protocol fields for IoT anomaly detection.
* **Nebhnani (2025) [10] (ProEn-XAI):** Couples hybrid deep learning with SHAP and LIME for zero-day attack identification.
* **Nandan and Das (2025) [11] (XAI4DDoS):** Uses TreeSHAP to inspect binary and multi-class DDoS signatures.
* **Haque et al. (2026) [17]:** Formulates XGBoost-SHAP pipelines to audit US critical infrastructure risks.
* **Berrezzek et al. (2026) [16]:** Deploys explainable hybrid feature selection in IoMT medical environments.
* *Crucial Limitation:* None of these works evaluate whether explanations can be used to audit synthetic data fidelity.

---

## 2. Comprehensive Taxonomy Matrix: 19 Prior Works vs. Our Proposed Working

| Ref / Authors | Year | Architecture | Balancing Strategy | Classifier Core | XAI Layer | Synthetic Data XAI Fidelity Validation? |
| :--- | :---: | :---: | :--- | :--- | :---: | :---: |
| **[1] Barnard et al.** | 2022 | Two-Stage | None | XGBoost + Autoencoder | SHAP | ❌ No |
| **[2] Mahmoud et al.** | 2025 | Two-Stage | None (Unweighted) | Decision Trees / RF | TreeSHAP | ❌ No |
| **[3] Yi-fan** | 2025 | Single-Stage | SC-CGAN (Flat) | Deep Neural Network | None | ❌ No |
| **[4] Ma et al.** | 2025 | Two-Stage | Cost-Sensitive | Dynamic Cost Learner | Shapley Values | ❌ No |
| **[5] Uddin et al.** | 2025 | Hierarchical | None | LightGBM / RF | None | ❌ No |
| **[6] Muhammad et al.** | 2025 | Single-Stage | SMOTE | Ensemble + Distillation | SHAP + LIME | ❌ No |
| **[7] El-Nashar et al.** | 2025 | Single-Stage | None | Random Forest / SVM | SHAP | ❌ No |
| **[8] Alatawi** | 2025 | Single-Stage | SMOTE | Ensemble (RF+XGB) | SHAP | ❌ No |
| **[9] Gupta et al.** | 2025 | Single-Stage | None | RF + XGBoost + LGBM | None | ❌ No |
| **[10] Nebhnani** | 2025 | Single-Stage | SMOTE | Hybrid CNN-LSTM | SHAP + LIME | ❌ No |
| **[11] Nandan & Das** | 2025 | Single-Stage | SMOTE | XGBoost / CatBoost | SHAP | ❌ No |
| **[12] Alabdulatif** | 2025 | Single-Stage | None | Ensemble Deep Learning | SHAP | ❌ No |
| **[13] Bibers et al.** | 2025 | Single-Stage | ROS | Ensemble Stacking | None | ❌ No |
| **[14] Zhang et al.** | 2023 | Two-Stage | None | LightGBM + Deep Net | None | ❌ No |
| **[15] Bedi et al.** | 2021 | Single-Stage | Siamese Margin | Siam-IDS Deep Metric | None | ❌ No |
| **[16] Berrezzek et al.** | 2026 | Single-Stage | None | Hybrid Trees | SHAP | ❌ No |
| **[17] Haque et al.** | 2026 | Single-Stage | Cost-Sensitive | XGBoost | TreeSHAP | ❌ No |
| **[18] Alonso & Pellicer** | 2026 | Single-Stage | CTGAN (Flat) | Random Forest | SHAP | ❌ No (Visual only) |
| **[19] Zhang et al.** | 2020 | Single-Stage | CWGAN-GP (Flat) | Cost-Sensitive SAE | None | ❌ No |
| **PROPOSED WORK** | **2026** | **Two-Stage** | **Attack-Only WGAN-GP** | **RF + XGBoost + LGBM** | **Stage-Wise SHAP** | **✅ YES (Novelty 1)** |

---

## 3. Core Architectural Differentiators: How Our Working Solves Prior Gaps

1. **Decoupled Two-Stage Gatekeeper vs. Monolithic Flattening:**
   Monolithic models suffer from extreme false positive alarms on benign traffic when pushed to capture rare attacks. Our framework deploys a Stage 1 Gatekeeper with threshold tuning ($\tau \le 0.10$) that isolates 97.74% of benign traffic with a false positive rate of only $0.0227$, achieving $99.98\%$ attack recall before Stage 2 classification.
2. **Attack-Conditioned WGAN-GP vs. Flat All-Class GANs:**
   Prior generative studies (Zhang et al. [19], Alonso et al. [18], Yi-fan [3]) train GANs across both normal and attack records. Because normal traffic represents $53\%$ to $90\%$ of training data, discriminator gradients are dominated by benign patterns, resulting in mode collapse for minority attacks (U2R/R2L). Our WGAN-GP is trained strictly on attack records, eliminating benign contamination and stabilizing Wasserstein convergence.
3. **Multi-Tree Soft-Voting Ensemble vs. Single Estimators:**
   While standalone models like XGBoost or LightGBM have individual failure modes, our Stage 2 deploys a soft-voting ensemble combining Random Forest (variance reduction via feature bagging), XGBoost (bias reduction via second-order Taylor tree boosting), and LightGBM (histogram leaf-wise efficiency). This delivers balanced multi-class discrimination across DoS, Probe, R2L, and U2R.
4. **Quantitative SHAP Explanation Fidelity (Novelty Claim 1):**
   Existing literature evaluates synthetic data solely through classification accuracy (e.g., TSTR). We introduce the first quantitative explanation fidelity framework evaluating Spearman rank correlation ($\rho_s$), top-10 Jaccard overlap ($J_{10}$), and cosine similarity ($S_C$) between synthetic and real attack attributions. We prove that WGAN-GP achieves $\rho_s = 0.881$ and $J_{10} = 0.750$, significantly outperforming SMOTE ($\rho_s = 0.784, J_{10} = 0.600$).
5. **Stage-Wise SHAP Deconvolution (Novelty Claim 2):**
   Monolithic SHAP explanations mix normal-boundary features with attack-discriminating features. Our stage-wise deconvolution cleanly separates Stage 1 features (volumetric flow attributes like `src_bytes`, `serror_rate`, `count`) from Stage 2 features (host-level exploit markers like `num_root`, `hot`, `is_guest_login`, `dst_host_srv_diff_host_rate`), providing security analysts with actionable, unambiguous forensic evidence.

---

## 4. Empirical Validation and Benchmark Results

Across 10 independent random seeds on NSL-KDD, UNSW-NB15, and CIC-IDS2017:
* **Stage 1 Normal FPR:** $0.0227$ (lowest false positive rate among all evaluated baselines).
* **Attack Filtering Recall:** $99.98\%$ at $\tau = 0.10$ with only 2 false negatives out of 11,726 validation attacks.
* **CIC-IDS2017 Multi-Class Accuracy:** $99.80\%$, Macro-F1: $0.8787$, Weighted-F1: $0.9978$, MCC: $0.9932$.
* **Online Inference Latency:** $0.0071$ ms/flow ($>140,800$ packets/second), guaranteeing line-rate edge deployment.
* **Memorization Audit:** $0.0\%$ nearest-neighbor memorization on U2R training samples, confirming genuine manifold generalization.

---

## 5. Publication Positioning & Reviewer Defense Strategy

| Potential Reviewer Concern | Technical Defense & Evidence from Paper |
| :--- | :--- |
| *"Two-stage IDS architectures have been proposed before (e.g., Tauscher et al., Mahmoud et al.). What is new?"* | **Defense:** We do not claim that two-stage architectures are fundamentally new. We explicitly cite prior two-stage literature. Our novelty lies in the synthesis: combining an attack-conditioned WGAN-GP oversampler strictly inside Stage 2, coupled with the first empirical demonstration of SHAP Explanation Fidelity (Claim 1) and Stage-Wise Explanation Deconvolution (Claim 2). |
| *"Why not use SMOTE instead of a computationally intensive WGAN-GP?"* | **Defense:** As demonstrated in Table IV, while SMOTE increases minority recall by linear interpolation, it distorts feature correlation manifolds, yielding lower SHAP attribution fidelity (Spearman $\rho_s = 0.784$ vs $0.881$ for WGAN-GP). In high-dimensional categorical spaces, SMOTE synthesizes invalid fractional states that violate domain constraints, whereas our WGAN-GP enforces categorical argmax consistency. |
| *"How does Stage 1 avoid dropping low-frequency attacks like U2R and R2L?"* | **Defense:** Standard binary models with default 0.5 thresholds drop rare attacks. Our paper provides an exhaustive validation threshold scan (Table II), proving that calibrating $\tau \le 0.10$ achieves $99.98\%$ recall on attacks, ensuring rare exploits pass safely into Stage 2. |

---

## 6. Complete Verified References (19 Papers with DOIs)

1. **[1]** P. Barnard, N. Marchetti, and L. A. DaSilva, "Robust Network Intrusion Detection Through Explainable Artificial Intelligence (XAI)," *IEEE Networking Letters*, vol. 4, no. 3, pp. 167–171, Sept. 2022, doi: [10.1109/LNET.2022.3186589](https://doi.org/10.1109/LNET.2022.3186589).
2. **[2]** M. M. Mahmoud, Y. Omar, and A. Abdel-Hamid, "XI2S-IDS: An Explainable Intelligent 2-Stage Intrusion Detection System," *Future Internet*, vol. 17, no. 1, art. 25, Jan. 2025, doi: [10.3390/fi17010025](https://doi.org/10.3390/fi17010025).
3. **[3]** Z. Yi-fan, "CSAGC-IDS: A Dual-Module Deep Learning Network Intrusion Detection Model for Complex and Imbalanced Data," *arXiv preprint arXiv:2505.14027*, May 2025, doi: [10.48550/arxiv.2505.14027](https://doi.org/10.48550/arxiv.2505.14027).
4. **[4]** X. Ma, L. Yang, and H. Wang, "Dynamic Feedback-Based Cost-Sensitive Learning for Imbalanced Intrusion Detection," *Proc. IEEE SMC*, pp. 2135–2142, Oct. 2025, doi: [10.1109/smc58881.2025.11343430](https://doi.org/10.1109/smc58881.2025.11343430).
5. **[5]** M. A. Uddin, N. H. Chu, R. Rafeh, and M. Barika, "A Scalable Hierarchical Intrusion Detection System for Internet of Vehicles," *IEEE Internet of Things Journal*, pp. 1–1, Jan. 2025, doi: [10.1109/jiot.2025.3590966](https://doi.org/10.1109/jiot.2025.3590966).
6. **[6]** A. Muhammad, J. Ahmad Mian, H. Bin Safayat, S. Houbing, and J. Zhan-peng, "xIDS-EnsembleGuard: An Explainable Ensemble Learning-based Intrusion Detection System," *TechRxiv*, Apr. 2025, doi: [10.13016/m2obje-mmxq](https://doi.org/10.13016/m2obje-mmxq).
7. **[7]** R. El-Nashar, G. I. Salama, and A. Tammam, "An Efficient Intrusion Detection for IoT Networks: A SHAP-Driven Machine Learning Approach," *Proc. IEEE ICICIS*, pp. 433–439, Nov. 2025, doi: [10.1109/icicis66182.2025.11313099](https://doi.org/10.1109/icicis66182.2025.11313099).
8. **[8]** M. N. Alatawi, "Enhancing Intrusion Detection Systems With Advanced Machine Learning Techniques: An Ensemble and Explainable Artificial Intelligence Approach," *Security and Privacy*, vol. 8, no. 1, art. e496, Jan. 2025, doi: [10.1002/spy2.496](https://doi.org/10.1002/spy2.496).
9. **[9]** H. Gupta, K. Singh, and A. K. Phulre, "Improving Intrusion Detection by using Multiclass Cyberattack Classification LightGBM, XGBoost, and Random Forest," *Proc. IEEE ICTBIG*, pp. 1–6, Dec. 2025, doi: [10.1109/ictbig68706.2025.11323736](https://doi.org/10.1109/ictbig68706.2025.11323736).
10. **[10]** N. Nebhnani, "ProEn-XAI: A High-Precision IDS Model for Zero-Day Attack Detection Using Hybrid Deep Learning and SHAP-LIME Interpretability," *IJCNC*, vol. 11, no. 12s, pp. 1339–1354, June 2025, doi: [10.64252/cpzyn974](https://doi.org/10.64252/cpzyn974).
11. **[11]** M. Nandan and R. Das, "XAI4DDoS: Explainable ML Pipelines for Binary and Multiclass Network Attack Detection," *Proc. IEEE INSPECT*, pp. 1–7, Nov. 2025, doi: [10.1109/inspect67393.2025.11350899](https://doi.org/10.1109/inspect67393.2025.11350899).
12. **[12]** A. Alabdulatif, "A Novel Ensemble of Deep Learning Approach for Cybersecurity Intrusion Detection with Explainable Artificial Intelligence," *Applied Sciences*, vol. 15, no. 14, art. 7984, July 2025, doi: [10.3390/app15147984](https://doi.org/10.3390/app15147984).
13. **[13]** I. Bibers et al., "Ensemble-IDS: An Ensemble Learning Framework for Enhancing AI-Based Network Intrusion Detection Tasks," *Applied Sciences*, vol. 15, no. 19, art. 10579, Sept. 2025, doi: [10.3390/app151910579](https://doi.org/10.3390/app151910579).
14. **[14]** H. Zhang, B. Zhang, L. Huang, Z. Zhang, and H. Huang, "An Efficient Two-Stage Network Intrusion Detection System in the Internet of Things," *Information*, vol. 14, no. 2, art. 77, Jan. 2023, doi: [10.3390/info14020077](https://doi.org/10.3390/info14020077).
15. **[15]** P. Bedi, N. Gupta, and V. Jindal, "I-SiamIDS: an improved Siam-IDS for handling class imbalance in network-based intrusion detection systems," *Applied Intelligence*, vol. 51, pp. 1133–1151, Sept. 2021, doi: [10.1007/s10489-020-01886-y](https://doi.org/10.1007/s10489-020-01886-y).
16. **[16]** A. Berrezzek, H. Djellali, G. Mallardi, and L. Mahnane, "Explainable Hybrid Feature Selection for Intrusion Detection in Internet of Medical Things Environments," *arXiv preprint arXiv:2608.00869*, Aug. 2026.
17. **[17]** B. M. T. Haque, Md. A. Rahman, Md. S. K. C. Rubel, and Md. I. Hossan, "Explainable AI-Driven Cyber Risk Analytics and Model Reliability Assessment for Intelligent Governance of U.S. Critical Infrastructure: An XGBoost and SHAP-Based Intrusion Detection Framework," *Engineering Reports*, June 2026, doi: [10.25163/engineering.2110762](https://doi.org/10.25163/engineering.2110762).
18. **[18]** J. L. V. Alonso and C. Pellicer, "Forensic-Oriented Intrusion Detection Using Synthetic Network Traffic Data and Explainable Artificial Intelligence," *arXiv preprint arXiv:2607.00763*, July 2026.
19. **[19]** G. Zhang, X. Wang, R. Li, Y. Song, J. He, and J. Lai, "Network Intrusion Detection Based on Conditional Wasserstein Generative Adversarial Network and Cost-Sensitive Stacked Autoencoder," *IEEE Access*, vol. 8, pp. 190431–190447, Oct. 2020, doi: [10.1109/ACCESS.2020.3031892](https://doi.org/10.1109/ACCESS.2020.3031892).
