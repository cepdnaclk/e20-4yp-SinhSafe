---
layout: home
permalink: index.html
repository-name: e20-4yp-SinhSafe
title: SinhSafe - Multi-Model Detection of Cyberbullying and Hate Speech
---

<p align="center">
<img src="images/cover_page.jpg" width="100%">
</p>

# SinhSafe: An Iterative Deep Learning & Ensemble Approach to Sinhala Harassment Detection

<p align="center">
  <img src="images/SinhSafe.png" width="200" />
</p>

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

<p align="center">
  <img src="images/basemodels.png"/>
</p>

---

## Experiment Setup and Implementation

### 1. Model Architectures
We engineered three distinct architectures, adding custom layers to prevent overfitting:

* **XLM-RoBERTa (Large):** Features a custom dense head with **20% Dropout** and **GELU activation** to manage the 1024-dimensional feature vector.
* **SinBERT (LSTM-Head):** Utilizes a **Bi-Directional LSTM** (512 units) with **Dual-Pooling** (Average + Max) to capture long-range dependencies in native Sinhala script.
* **SinLLaMA (8B):** An instruction-tuned LLM using **4-bit NF4 Quantization (QLoRA)** and **LoRA** adapters for parameter-efficient tuning.

### 2. Training Strategies, Hyperparameter Search & Loss Analysis
To find the perfect training arguments (learning rate, batch size, weight decay) and prevent overfitting, we tested 12 distinct iterations across our three architectures. We tracked training vs. evaluation loss for every version and compared their overall metrics to select the "Best-Fit" model for our final ensemble.

#### A. SinBERT (LSTM-Head) - 5 Versions Tested
We utilized **5-Fold Stratified Cross-Validation** to isolate the best-performing epoch and evaluate stability across different data splits.

<table>
  <tr>
    <td align="center">
      <strong>Version 1</strong><br>
      <img src="images/sinbert/SinBERT_Version_1_Loss_Curve.png" alt="SinBERT V1">
    </td>
    <td align="center">
      <strong>Version 2</strong><br>
      <img src="images/sinbert/SinBERT_Version_2_Loss_Curve.png" alt="SinBERT V2">
    </td>
  </tr>
  <tr>
    <td align="center">
      <strong>Version 3</strong><br>
      <img src="images/sinbert/SinBERT_Version_3_Loss_Curve.png" alt="SinBERT V3">
    </td>
    <td align="center">
      <strong>Version 4</strong><br>
      <img src="images/sinbert/SinBERT_Version_4_Loss_Curve.png" alt="SinBERT V4">
    </td>
  </tr>
  <tr>
    <td colspan="2" align="center">
      <strong>Version 5</strong><br>
      <img src="images/sinbert/SinBERT_Version_5_Loss_Curve.png" alt="SinBERT V5" style="width: 50%;">
    </td>
  </tr>
</table>

**SinBERT Performance Comparison**
<p align="center">
  <img src="images/sinbert/SinBERT_Metrics_Comparison.png" alt="SinBERT Metric Comparison" style="width: 80%;">
</p>
<br>

<h3 style="margin-top: 20px;">Rationale for Selecting Version 4</h3>

Based on the comprehensive metric comparison and loss curve analysis, we selected **Version 4** as our optimal SinBERT model. While Versions 2 and 3 showed marginally higher raw accuracy, Version 4 demonstrated an exceptionally strong **Weighted Precision**, which is critical in moderation systems to minimize **"False Positives"** (unfairly penalizing normal users). 

Furthermore, the V4 Loss Curve presented a mathematically perfect early-stopping threshold: the evaluation loss hit a distinct, sharp minimum exactly at **Epoch 2**. By halting training at this exact checkpoint, we captured the model at its absolute peak generalization, completely avoiding the severe overfitting observed in the later epochs of the other versions.

<br>

**🏆 Winning Parameters for Production Model(SinBERT Best Version):**
```text
max_len       : 128
batch_size    : 16
epochs        : 2
learning_rate : 2e-05
dropout_p     : 0.3
```
#### B. XLM-RoBERTa (Large) - 3 Versions Tested
Similar to SinBERT, XLM-R was evaluated using Stratified 5-Fold Cross-Validation, relying on the lowest evaluation loss and highest weighted precision to prevent data leakage.

<table>
  <tr>
    <td align="center">
      <strong>Version 1</strong><br>
      <img src="images/xlm/XLMR_Version_1_Loss_Curve.png" alt="XLM-R V1">
    </td>
    <td align="center">
      <strong>Version 2</strong><br>
      <img src="images/xlm/XLMR_Version_2_Loss_Curve.png" alt="XLM-R V2">
    </td>
  </tr>
  <tr>
    <td colspan="2" align="center">
      <strong>Version 3</strong><br>
      <img src="images/xlm/XLMR_Version_3_Loss_Curve.png" alt="XLM-R V3" style="width: 50%;">
    </td>
  </tr>
</table>

**XLM-RoBERTa Performance Comparison**
<p align="center">
  <img src="images/xlm/XLMR_Metrics_Comparison.png" alt="XLM-R Metric Comparison" style="width: 80%;">
</p>

<br>

<h3 style="margin-top: 20px;">Rationale for Selecting Version 2</h3>

Based on the multi-version benchmark, we selected **Version 2** as the production-ready model for the XLM-RoBERTa architecture. Version 2 achieved the highest **F1-Score (80.41%)** and **Accuracy (80.46%)** across all tested iterations. 

While the loss curves indicate that Version 2 eventually began to overfit as training progressed, our implementation of **Early Stopping** allowed us to capture the model weights at the optimal convergence point (Epoch 3). This balanced peak performance with sufficient generalization to handle the linguistic variance in our large-scale unlabeled dataset.

<br>

**🏆 Winning Parameters for Production Model(XLM-R Best Version):**
```text
num_train_epochs : 10
batch_size       : 32
learning_rate    : 2e-05
warmup_steps     : 500
weight_decay     : 0.01
```
#### C. SinLLaMA (8B) - 4 Versions Tested
For the Generative LLM, we evaluated via an 80/10/10 Train/Val/Test split. To prevent the "Testing Collapse" caused by memorization, we implemented strict **Early Stopping**: training halted if evaluation loss increased for 3 consecutive intervals (every 50 steps), capturing the checkpoint with the absolute lowest evaluation loss.

<table>
  <tr>
    <td align="center">
      <strong>Version 1</strong><br>
      <img src="images/siinllama/SinLLaMA_Version_1_Loss_Curve.png" alt="SinLLaMA V1">
    </td>
    <td align="center">
      <strong>Version 2</strong><br>
      <img src="images/siinllama/SinLLaMA_Version_2_Loss_Curve.png" alt="SinLLaMA V2">
    </td>
  </tr>
  <tr>
    <td align="center">
      <strong>Version 3</strong><br>
      <img src="images/siinllama/SinLLaMA_Version_3_Loss_Curve.png" alt="SinLLaMA V3">
    </td>
    <td align="center">
      <strong>Version 4</strong><br>
      <img src="images/siinllama/SinLLaMA_Version_4_Loss_Curve.png" alt="SinLLaMA V4">
    </td>
  </tr>
</table>

**SinLLaMA Performance Comparison**
<p align="center">
  <img src="images/siinllama/SinLLaMA_Metrics_Comparison.png" alt="SinLLaMA Metric Comparison" style="width: 80%;">
</p>

<br>

<h3 style="margin-top: 20px;">Rationale for Selecting Version 3</h3>

For the SinLLaMA architecture, **Version 3** was selected as the optimal configuration. This version achieved the highest overall performance metrics, specifically reaching a peak **F1-Score of 65.66%**. 

While the 8B parameter model exhibited a high tendency to overfit (as seen in the diverging loss curves of other versions), Version 3 maintained a more stable evaluation loss across training steps. By leveraging the specific hyperparameters of this iteration, we were able to maximize the generative potential of the model for our pseudo-labeling engine while mitigating the "Memorization Trap" common in large-scale instruction tuning.
<br>

**🏆 Winning Parameters for Production Model(SinLLaMA Best Version):**
```text
max_length        : 512
batch_size        : 16
num_train_epochs  : 1
learning_rate     : 5e-05
weight_decay      : 0.05
bf16              : True
```


### 3. Synthesizing V1 Production Models
After identifying the optimal hyperparameters and the exact "best-fit" epoch for each architecture, we moved out of the cross-validation phase. We retrained **XLM-RoBERTa**, **SinBERT**, and **SinLLaMA** on **100% of the V1 Dataset (6,075 documents)** using these winning parameters. This maximized the models' knowledge retention, resulting in three highly robust, inference-ready "V1 Production Models."

---

## The Ensemble Pseudo-Labeling Engine (V1 to V2)

To overcome data scarcity, we deployed these three V1 Production Models on 145,000 unlabelled social media comments. We applied a **Strict Extraction Logic** to build our final V2 Dataset:
1. **Direct Extraction:** Any label where at least one model had **>90% confidence**.
2. **Consensus Extraction:** Confidence between **80-90%** where XLM-R and SinBERT agreed.
3. **Manual Review:** Confidence between **40-80%** where all three models agreed; these were manually verified before inclusion.

This process allowed us to extend the Harassment class to 5,515 documents, creating a perfectly balanced V2 dataset (16,545 documents total) for final production training.

---

## Results and Analysis

The transition to the V2 dataset resulted in a massive performance leap across all architectures.

| Model | Parameter Size | V1 F1-Score | V2 F1-Score |
| :--- | :--- | :--- | :--- |
| **SinBERT** | ~110 Million | 77.9% | **90.7%** |
| **XLM-R** | ~550 Million | 80.4% | **86.9%** |
| **SinLLaMA** | ~8 Billion | 55.7% | 64.9% |

<table>
  <tr>
    <td align="center">
      <strong>V1 to V2 Performance Leap</strong><br>
      <img src="images/final_f1_leap.png" alt="F1 Score Leap">
    </td>
    <td align="center">
      <strong>Production Models: Final Eval Loss</strong><br>
      <img src="images/final_eval_loss.png" alt="Final Evaluation Loss">
    </td>
  </tr>
</table>

### Optimal Epoch & Loss Curves
By tracking training and evaluation loss, we successfully identified the best epoch to run our 100% data training without overfitting or underfitting.

<table>
  <tr>
    <td align="center">
      <strong>SinBERT Production Model</strong><br>
      <img src="images/SinBERT_Best_Version_Loss_Curve.png" alt="SinBERT Best Production Curve">
    </td>
    <td align="center">
      <strong>XLM-RoBERTa Production Model</strong><br>
      <img src="images/XLM_Best_Version_Loss_Curve.png" alt="XLM-R Best Production Curve">
    </td>
  </tr>
</table>

### The "LLM Memorization Trap"
A critical discovery was the failure of SinLLaMA to generalize. Despite its 8B parameters, it exhibited **Severe Overfitting**, crashing to 64.9% on unseen test data, whereas the lightweight Encoders (SinBERT/XLM-R) learned general linguistic rules more effectively. 

<p align="center">
  <img src="images/sinllama_memorization_trap.png" alt="SinLLaMA Memorization Trap Graph" style="width: 80%;">
  <br>
  <em>Figure: Visualizing the divergence between training and evaluation loss, indicating a collapse in generalization.</em>
</p>

---

## Conclusion
The final **SinhSafe Production Ensemble** utilizes **Soft-Voting (Probability Averaging)** between XLM-RoBERTa and SinBERT. This configuration provides a culturally aware, real-time moderation solution that outperforms traditional baselines while avoiding the massive computational overhead of generative LLMs.

## Links
- [Project Repository](https://github.com/cepdnaclk/e20-4yp-SinhSafe)
- [Project Demo Video](https://cepdnaclk.github.io/e20-4yp-SinhSafe)
- [Department of Computer Engineering](http://www.ce.pdn.ac.lk/)
