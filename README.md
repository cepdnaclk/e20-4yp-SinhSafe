# SinhSafe: A Deep Learning Approach to Sinhala Harassment Detection

**SinhSafe** is a research project focused on detecting harmful language in code-mixed Sinhala–English ("Singlish") text. Rather than a binary flag/no-flag system, SinhSafe uses a **ternary classifier** that separates ordinary colloquial speech from crude-but-harmless language and genuinely targeted harassment — closing the semantic gap that causes traditional moderation systems to over-censor normal speech or under-flag real harm.

This repository hosts the dataset preprocessing scripts, annotation protocol, and training pipeline for our benchmarked models: traditional ML baselines, transformer encoders (SinBERT, XLM-RoBERTa), and a generative LLM (SinLLaMA-8B), plus the soft-voting ensemble of the two best encoders.

## 👤 Author Information
* **Author:** P.D. Thilakasiri (E/20/397)
* **Co-authors / Supervisors:** Dr. Eng. Damayanthi Herath, Dr. Eng. Sampath Deegalla
* **Institution:** University of Peradeniya, Sri Lanka
* **Contact:** e20397@eng.pdn.ac.lk

## 📌 Project Overview
* **Goal:** Build a robust ternary classifier that accurately separates **Normal**, **Offensive**, and **Harassment** content in code-mixed Sinhala text.
* **Approach:** Benchmark across the full spectrum of NLP methods — traditional ML, bidirectional transformer encoders, and a generative LLM — then determine whether combining the best models actually improves on the best single model, using statistically rigorous evaluation (bootstrap confidence intervals), not just raw score comparison.

### Classification Classes (Ternary)
1. **Normal:** Everyday, non-offensive speech and colloquial slang.
2. **Offensive:** Rude or crude language without targeted intent to harm.
3. **Harassment:** Targeted, harmful content aimed at an individual or group.

## 🧪 Methodology & Architecture

### 1. Data Engineering & Preprocessing
* **Source:** 6,075 documents sourced from SOLD (Ranasinghe et al., 2022) — the Sinhala Offensive Language Dataset.
* **Cleaning & Transliteration:** Stripped noise (e.g. `@user`); Singlish → Sinhala via the Google Translate API.
* **Annotation & QC:** 3 annotators labeled independently; inter-annotator agreement measured; majority voting used to resolve disagreements.
* **Final Split:** Stratified 90% train / 10% held-out test (random state fixed at 42 for full reproducibility). Class distribution is imbalanced — roughly 38% Normal, 33% Offensive, 28% Harassment (minority class) — so stratified sampling is used for every split and every cross-validation fold.

### 2. Three Modeling Approaches
Benchmarked in order of increasing complexity (Occam's Razor: only add complexity that earns better held-out performance):

| Family | Models |
| :--- | :--- |
| Traditional ML | Naïve Bayes, Logistic Regression, Linear SVM, Random Forest, MLP |
| Transformer Encoders (bidirectional, non-generative) | SinBERT Large (~110M), XLM-RoBERTa Large (~550M), each with a Bi-LSTM / GELU classification head |
| Generative LLM (decoder-only, autoregressive) | SinLLaMA-8B, fine-tuned with QLoRA + Alpaca-style prompting |

### 3. Architecture Optimization
* 12 model versions engineered on the 90% training partition (5 SinBERT, 3 XLM-RoBERTa, 4 SinLLaMA).
* Stratified 5-fold CV for SinBERT & XLM-RoBERTa; stratified 80/10 split within the training partition for SinLLaMA (full CV wasn't practical at 8B parameters).
* Early stopping on eval-loss for all three, to select the best epoch before the training/validation loss curves diverge (overfitting).

### 4. Ensemble
Final ensemble is a **soft-voting combination of SinBERT + XLM-RoBERTa Large's predicted probabilities** — not a 3-model pseudo-labeling scheme. SinLLaMA was excluded from the ensemble due to its poor generalization (see below).

## 📊 Results

Evaluated as **macro F1** on the 10% held-out test set (best configs, trained on 100% of the 90% partition):

| Model | Macro F1 | 95% CI (bootstrap, 1,000 iterations) |
| :--- | :--- | :--- |
| Traditional ML baselines | ~65% (plateau) | — |
| SinBERT Large | 73.44% | [69.83%, 77.02%] |
| XLM-RoBERTa Large | 75.44% | [71.95%, 78.89%] |
| SinLLaMA-8B | 55.7% | — (severe overfitting) |
| **SinhSafe Ensemble** | **76.21%** | [72.96%, 79.66%] |

**Is the ensemble actually better?** The ensemble's advantage over standalone XLM-RoBERTa is only +0.89% (95% CI: [-1.61%, 3.36%]) — the interval crosses zero, so it is **not statistically significant**. A well-tuned standalone XLM-RoBERTa performs comparably to the costlier ensemble.

### Why Did the LLM Fail?
SinLLaMA-8B achieved high training accuracy but collapsed to 55.7% macro F1 on unseen data — a training/validation loss divergence indicating the model memorized training examples rather than learning generalizable patterns, likely because the fine-tuning data was small relative to the model's 8B-parameter scale. **Takeaway:** in data-scarce, low-resource NLP settings, encoder-only transformers with dedicated classification heads generalize far more reliably than large generative models fine-tuned on the same limited data — bigger isn't always better.

## 🚀 Proposed Deployment: Human-in-the-Loop Pipeline
Not an automatic censor — a decision-support pipeline:
1. Standalone XLM-RoBERTa scores incoming content (Normal / Offensive / Harassment).
2. Low-confidence predictions are routed to human moderators, not auto-actioned.
3. Human-reviewed and high-confidence cases are fed back into the dataset.
4. The model is periodically retrained, improving accuracy and fairness over time.

**Dataset Annotation Acknowledgements**

The development of the SinhSafe benchmark relied on the creation of two distinct datasets, specifically annotated to capture both male and female points of view (POV) regarding online harassment and cyberbullying. Sincere gratitude is extended to the following individuals for their dedicated efforts in manually classifying the code-mixed Sinhala-English corpus:

**Female POV Dataset Annotators**
* **Uduvita Arachchilage Dineeshi Thilakshika Dassanayaka** – Faculty of Applied Sciences, Sabaragamuwa University of Sri Lanka
* **Prabodhani Buddhimali Ranawaka** – Faculty of Medicine, Wayamba University of Sri Lanka

**Male POV Dataset Annotators**
* **Priyankarage Dimantha Thilakasiri** – University of Peradeniya
* **G.D.D. Kumaranathunga** – University of Peradeniya
* **R. Priyadarshana** – University of Peradeniya
