# CTCVR-Prediction-for-BBA-Buyer-Book-Activity-

An end-to-end Click-Through Conversion Rate (CTCVR) prediction pipeline modeling user purchase behavior in e-commerce ecosystems. This project reproduces and evaluates the core architecture framework proposed by **Wei et al. (2024)** in *Expert Systems with Applications* (Vol. 238).

[![DOI](https://shields.io)](https://doi.org/10.1016/j.eswa.2023.122276)
[![Python](https://shields.io)](https://python.org)
[![Framework](https://shields.io)](https://scikit-learn.org)

---

## 📌 Project Overview

Predicting click-through conversion rates is a core component of modern recommendation engines and e-commerce platforms. This pipeline implements a deep hybrid modeling paradigm designed around three central pillars of an e-commerce ecosystem: **Buyer, Book, and Activity (BBA)**. 

The implementation features a custom mathematical representation of feature interactions coupled with deep neural learning layers, benchmarked directly against advanced gradient-boosted trees and original paper targets.

---

## 🏗️ Architecture Design

The framework processes sparse user-product-activity graphs and tabular profiles through a parallel multi-channel pipeline:

1. **Factorization Machine (FM Channel):** 
   - A custom scratch-built NumPy implementation capturing 1st-order linear traits and 2nd-order non-linear feature interactions efficiently via the \(O(k \cdot d)\) time-complexity optimization trick.
2. **Deep Neural Network (DNN Channel):** 
   - A fully connected multi-layer perceptron configuration (2 × 40 hidden layers) utilizing Rectified Linear Unit (ReLU) activations to model higher-order non-linear semantic spaces.
3. **BBA2vec Representation Layer:** 
   - Tripartite graph embedding sequence mimicking structural representation learning. Generates 24-dimensional continuous feature representations via separate `TruncatedSVD` matrices extracted over user × product and user × activity co-occurrence patterns (restricted strictly to training splits to ensure zero data leakage).
4. **Weighted Ensemble Blending:** 
   - Combines downstream probability outputs via optimized validation-set grid sweeps.

---

## 📊 Feature Engineering Pack

The pipeline engineers high-utility explicit features to bridge the performance gap between synthetic data baselines and commercial production environments:

* **Temporal Dynamics:** Extracted granular time features (`click_hour`, `click_dayofweek`, `click_month`, `click_year`) combined with a custom exponential time-decay proxy feature \(e^{-\lambda \cdot t}\) based on campaign duration recency (Section 3.4 of the paper).
* **Behavior Interaction Profiles:** Historical CTR estimation tracking consumer interaction ratios (`historical_purchase_count / historical_click_count + 1`).
* **Value Signals:** Compounded numeric signals computing price-to-rating proportions and exact mathematical discount weights.
* **Non-Linear Intensity Probes:** Multi-dimensional interactions including logarithmic behavior histories, user activity frequencies relative to profile age, and historical tracking signals (`rating_x_reviews`, `ctr_x_time_decay`).

---

## 📈 Metric Reference Targets

Performance is rigorously analyzed against the official publication goals over an **8:1:1 split** on a 50,000-sample UPU (User-Product-Usage) dataset:

| Target Metric | Target Value |
| :--- | :--- |
| **ROC-AUC** | `0.9021` |
| **Log Loss** | `0.3838` |
| **F1-Score** | `0.7900` |

---

## 🚀 Getting Started

### Prerequisites

Ensure you have Python installed alongside the following primary computational dependencies:

```bash
pip install numpy pandas scikit-learn lightgbm scipy matplotlib seaborn
```

### File Hierarchy & Local Setup

1. Change your dataset directory configurations within the main notebook script to match your native local folder paths:
   ```python
   SAVE_DIR  = r"yourdrive:\yourfolder"
   DATA_PATH = os.path.join(SAVE_DIR, "ctcvr_dataset.csv")
   ```
2. Execute the notebook or Python script sequentially to run data processing, tripartite SVD building, network training, and automated visualization exports.

---

## 📉 Visualized Artifact Outputs

Upon code compilation, the script automatically builds and saves several diagnostic plots to your target working directory:
* `ml_01_roc_curves.png`: Multi-model Receiver Operating Characteristic comparisons mapping true/false positive tradeoffs against paper benchmarks.
* `ml_02_pr_curves.png`: Precision-Recall sweeps capturing optimal operational performance.
* `ml_03_model_comparison.png`: Structured triple-bar metric charts comparing baseline configurations, BBA2vec-infused frameworks, and targeted paper achievements.
* `ml_04_confusion_matrix.png`: Heatmap breakdown summarizing localized classification matrices.
* `ml_05_score_distribution.png`: Probability density mapping displaying class-separation stability.
* `ml_06_feature_importance.png`: Feature weight ranking derived from gradient-boosted decision trees.
* `ml_07_dnn_training_curve.png`: Convergence validation logging loss declines versus model iterations.

---

## 📜 Citation Reference

If utilizing this codebase or structural framework for academic or industrial research, please credit the original authors:

```bibtex
@article{wei2024ctcvr,
  title={Click-through conversion rate prediction model of book e-commerce platform based on feature combination and representation},
  author={Wei, Shihong and Yang, Zhou and Zhang, Jian and Zeng, Yang and Li, Qian and Xiao, Yunpeng},
  journal={Expert Systems with Applications},
  volume={238},
  pages={122276},
  year={2024},
  publisher={Elsevier},
  doi={https://doi.org/10.1016/j.eswa.2023.122276}
}
