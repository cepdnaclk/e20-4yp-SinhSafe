---
layout: home
permalink: index.html
repository-name: e20-4yp-SinhSafe
title: SinhSafe - Multi-Model Detection of Cyberbullying and Hate Speech
---

![SinhSafe Project Banner](./images/cover_page.jpg)

# SinhSafe: An Iterative Deep Learning & Ensemble Approach to Sinhala Harassment Detection

![SinhSafe Thumbnail](./images/SinhSafe.png){: style="display:block; margin-left:auto; margin-right:auto; width:200px;"}

#### Team
- e20397, Thilakasiri P.D., [email](mailto:e20397@eng.pdn.ac.lk)

#### Supervisors
- Dr. Eng. Sampath Deegalla, [email](mailto:sampath@eng.pdn.ac.lk)

---

## Project Summary
SinhSafe is a high-precision content moderation framework designed for the linguistic complexities of Sinhala and code-mixed Singlish. Traditional moderation tools often fail on local languages due to the "Semantic Gap"—the difficulty in distinguishing between general vulgarity (Offensive) and targeted, malicious attacks (Harassment). 

This project addresses these challenges through a dual-phase iterative approach. We established a rigorous ground truth of ~4,000 manually annotated documents using Inter-Annotator Agreement (IAA). Finding that traditional ML baselines were capped at a ~65% F1-score, we engineered an ensemble of deep learning architectures: **XLM-RoBERTa (Large)**, **SinBERT**, and **SinLLaMA**. By deploying these models in a **3-Model Ensemble Pseudo-Labeling Engine**, we tripled our dataset size to a perfectly balanced V2 corpus of 16,545 documents. Our final production system utilizes a soft-voting ensemble of the encoder models, achieving a peak F1-score of **90.7%** while maintaining real-time inference efficiency.

---

## Methodology & The Data Engine

### 1. The Data Pipeline
The SinhSafe pipeline begins with raw social media ingestion followed by a hybrid preprocessing engine:
* **Noise Removal:** Custom scripts to strip handles (e.g., @user) and social media artifacts.
* **Transliteration:** Integration of high-accuracy Singlish-to-Sinhala conversion.
* **Manual Annotation:** Establishing a baseline "Gold Standard" using strict rule sets for Harassment, Offensive, and Normal categories.

### 2. Baseline Comparison
Before moving to Deep Learning, we evaluated our V1 dataset against traditional algorithms:
* **Tested Models:** Naive Bayes, Linear SVM, Random Forest, Logistic Regression, and MLP.
* **The "F1 Ceiling":** All traditional models failed to exceed a 65% F1-Score, proving that semantic nuance in code-mixed text requires transformer-based architectures.

> **[PLACEHOLDER: Baseline_Performance_Comparison_Graph.png]**

---

## Experiment Setup and Implementation

### 1. Model Architectures
We engineered three distinct architectures, adding custom layers to prevent overfitting:

* **XLM-RoBERTa (Large):** Features a custom dense head with **20% Dropout** and **GELU activation** to manage the 1024-dimensional feature vector.
* **SinBERT (LSTM-Head):** Utilizes a **Bi-Directional LSTM** (512 units) with **Dual-Pooling** (Average + Max) to capture long-range dependencies in native Sinhala script.
* **SinLLaMA (8B):** An instruction-tuned LLM using **4-bit NF4 Quantization (QLoRA)** and **LoRA** adapters for parameter-efficient tuning.

### 2. Training Strategies & Hyperparameter Search
To find the perfect training arguments (learning rate, batch size, weight decay), we tested multiple iterations of each architecture. 

> **[PLACEHOLDER: Hyperparameter_Versions_Comparison_Graph.png]**

* **Validation:** We used **5-Fold Stratified Cross-Validation** for encoder models, selecting the "Best Fold" based on high weighted precision and lowest evaluation loss.
* **Early Stopping:** For SinLLaMA, we implemented a patience-based stop (3 consecutive intervals of 50 steps) to recall the lowest evaluation loss checkpoint, preventing "Testing Collapse".

---

## The Ensemble Pseudo-Labeling Engine (V1 to V2)

To overcome data scarcity, we deployed our "V1 Production Models" on 145,000 unlabelled social media comments. We applied a **Strict Extraction Logic** to build our final V2 Dataset:
1. **Direct Extraction:** Any label where at least one model had **>90% confidence**.
2. **Consensus Extraction:** Confidence between **80-90%** where XLM-R and SinBERT agreed.
3. **Manual Review:** Confidence between **40-80%** where all three models agreed; these were manually verified before inclusion.

This process allowed us to extend the Harassment class to 5,515 documents, creating a perfectly balanced dataset for final production training.

---

## Results and Analysis

The transition to the V2 dataset resulted in a massive performance leap across all architectures.

| Model | Parameter Size | V1 F1-Score | V2 F1-Score |
| :--- | :--- | :--- | :--- |
| **SinBERT** | ~110 Million | 77.9% | **90.7%** |
| **XLM-R** | ~550 Million | 80.4% | **86.9%** |
| **SinLLaMA** | ~8 Billion | 55.7% | 64.9% |

> **[PLACEHOLDER: V1_vs_V2_Performance_Comparison_Graph.png]**

### Optimal Epoch & Loss Curves
By tracking training and evaluation loss, we successfully identified the best epoch to run our 100% data training without overfitting or underfitting.

> **[PLACEHOLDER: SinBERT_Best_Version_Loss_Curve.png]**

> **[PLACEHOLDER: XLM-R_Best_Version_Loss_Curve.png]**

### The "LLM Memorization Trap"
A critical discovery was the failure of SinLLaMA to generalize. Despite its 8B parameters, it exhibited **Severe Overfitting**, crashing to 64.9% on unseen test data, whereas the lightweight Encoders (SinBERT/XLM-R) learned general linguistic rules more effectively. 

> **[PLACEHOLDER: SinLLaMA_Testing_Collapse_Loss_Curve.png]**

---

## Conclusion
The final **SinhSafe Production Ensemble** utilizes **Soft-Voting (Probability Averaging)** between XLM-RoBERTa and SinBERT. This configuration provides a culturally aware, real-time moderation solution that outperforms traditional baselines while avoiding the massive computational overhead of generative LLMs.

## Links
- [Project Repository](https://github.com/cepdnaclk/e20-4yp-SinhSafe)
- [Project Demo Video](https://cepdnaclk.github.io/e20-4yp-SinhSafe)
- [Department of Computer Engineering](http://www.ce.pdn.ac.lk/)
