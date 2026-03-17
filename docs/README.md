---
layout: home
permalink: index.html
repository-name: e20-4yp-SinhSafe
title: SinhSafe - Multi-Model Detection of Cyberbullying and Hate Speech
---

![SinhSafe Project Banner](./images/cover_page.jpg)

# SinhSafe: An Iterative Deep Learning & Ensemble Approach to Sinhala Harassment Detection

<div align="center">
    <img src="./images/SinhSafe.png" style="width:200px;">
</div>

#### Team
- e20397, Thilakasiri P.D., [email](mailto:e20397@eng.pdn.ac.lk)

#### Supervisors
- Dr. Eng. Sampath Deegalla, [email](mailto:sampath@eng.pdn.ac.lk)

---

## Project Summary
[cite_start]SinhSafe is a high-precision content moderation framework designed for the linguistic complexities of Sinhala and code-mixed Singlish[cite: 3, 41]. [cite_start]Traditional moderation tools often fail on local languages due to the "Semantic Gap"—the difficulty in distinguishing between general vulgarity (Offensive) and targeted, malicious attacks (Harassment)[cite: 39, 44]. 

[cite_start]This project addresses these challenges through a dual-phase iterative approach[cite: 4]. [cite_start]We established a rigorous ground truth of ~4,000 manually annotated documents using Inter-Annotator Agreement (IAA)[cite: 72, 87]. [cite_start]Finding that traditional ML baselines were capped at a ~65% F1-score, we engineered an ensemble of deep learning architectures: **XLM-RoBERTa (Large)**, **SinBERT**, and **SinLLaMA**[cite: 93, 168]. [cite_start]By deploying these models in a **3-Model Ensemble Pseudo-Labeling Engine**, we tripled our dataset size to a perfectly balanced V2 corpus of 16,545 documents[cite: 176, 183]. [cite_start]Our final production system utilizes a soft-voting ensemble of the encoder models, achieving a peak F1-score of **90.7%** while maintaining real-time inference efficiency[cite: 181, 201, 296].

---

## Methodology & The Data Engine

### 1. The Data Pipeline
[cite_start]The SinhSafe pipeline begins with raw social media ingestion followed by a hybrid preprocessing engine[cite: 70, 71]:
* [cite_start]**Noise Removal:** Custom scripts to strip handles (e.g., @user) and social media artifacts[cite: 86].
* [cite_start]**Transliteration:** Integration of high-accuracy Singlish-to-Sinhala conversion[cite: 76, 86].
* [cite_start]**Manual Annotation:** Establishing a baseline "Gold Standard" using strict rule sets for Harassment, Offensive, and Normal categories[cite: 64, 87].

### 2. Baseline Comparison
[cite_start]Before moving to Deep Learning, we evaluated our V1 dataset against traditional algorithms[cite: 91, 92]:
* [cite_start]**Tested Models:** Naive Bayes, Linear SVM, Random Forest, Logistic Regression, and MLP[cite: 111, 112, 113, 115, 121].
* [cite_start]**The "F1 Ceiling":** All traditional models failed to exceed a 65% F1-Score, proving that semantic nuance in code-mixed text requires transformer-based architectures[cite: 93, 94].

> **[PLACEHOLDER: Baseline_Performance_Comparison_Graph.png]**

---

## Experiment Setup and Implementation

### 1. Model Architectures
[cite_start]We engineered three distinct architectures, adding custom layers to prevent overfitting:

* [cite_start]**XLM-RoBERTa (Large):** Features a custom dense head with **20% Dropout** and **GELU activation** to manage the 1024-dimensional feature vector[cite: 130, 131, 132].
* **SinBERT (LSTM-Head):** Utilizes a **Bi-Directional LSTM** (512 units) with **Dual-Pooling** (Average + Max) to capture long-range dependencies in native Sinhala script[cite: 141, 143].
* [cite_start]**SinLLaMA (8B):** An instruction-tuned LLM using **4-bit NF4 Quantization (QLoRA)** and **LoRA** adapters for parameter-efficient tuning[cite: 153, 154].

### 2. Training Strategies
* [cite_start]**Validation:** We used **5-Fold Stratified Cross-Validation** for encoder models, selecting the "Best Fold" based on high weighted precision and lowest evaluation loss[cite: 137, 163].
* **Early Stopping:** For SinLLaMA, we implemented a patience-based stop (3 consecutive intervals of 50 steps) to recall the lowest evaluation loss checkpoint, preventing "Testing Collapse"[cite: 158, 165, 234].

---

## The Ensemble Pseudo-Labeling Engine (V1 to V2)

To overcome data scarcity, we deployed our "V1 Production Models" on 145,000 unlabelled social media comments[cite: 176, 178]. We applied a **Strict Extraction Logic** to build our final V2 Dataset[cite: 179]:
1. **Direct Extraction:** Any label where at least one model had **>90% confidence**[cite: 180].
2. **Consensus Extraction:** Confidence between **80-90%** where XLM-R and SinBERT agreed[cite: 181].
3. **Manual Review:** Confidence between **40-80%** where all three models agreed; these were manually verified before inclusion[cite: 182].

This process allowed us to extend the Harassment class to 5,515 documents, creating a perfectly balanced dataset for final production training[cite: 183].

---

## Results and Analysis

The transition to the V2 dataset resulted in a massive performance leap across all architectures[cite: 191].

| Model | Parameter Size | V1 F1-Score | V2 F1-Score |
| :--- | :--- | :--- | :--- |
| **SinBERT** | ~110 Million | 77.9% | **90.7%** |
| **XLM-R** | ~550 Million | 80.4% | **86.9%** |
| **SinLLaMA** | ~8 Billion | 55.7% | 64.9% |

> **[PLACEHOLDER: V1_vs_V2_Performance_Comparison_Graph.png]**
> **[PLACEHOLDER: Training_Loss_Curves_All_Models.png]**

### The "LLM Memorization Trap"
[cite_start]A critical discovery was the failure of SinLLaMA to generalize[cite: 237]. [cite_start]Despite its 8B parameters, it exhibited **Severe Overfitting**, crashing to 64.9% on unseen test data, whereas the lightweight Encoders (SinBERT/XLM-R) learned general linguistic rules more effectively[cite: 234, 235].

---

## Conclusion
[cite_start]The final **SinhSafe Production Ensemble** utilizes **Soft-Voting (Probability Averaging)** between XLM-RoBERTa and SinBERT[cite: 277, 281]. [cite_start]This configuration provides a culturally aware, real-time moderation solution that outperforms traditional baselines while avoiding the massive computational overhead of generative LLMs[cite: 295, 297].

## Links
- [Project Repository](https://github.com/cepdnaclk/e20-4yp-SinhSafe)
- [Project Demo Video](https://cepdnaclk.github.io/e20-4yp-SinhSafe)
- [Department of Computer Engineering](http://www.ce.pdn.ac.lk/)
