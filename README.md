# SinhSafe: Multi-Model Detection of Cyberbullying and Hate Speech

**SinhSafe** is a research project focused on detecting abusive language in the Sinhala and Singlish (Sinhala written in English script) domains. Unlike simple binary classification systems, this project distinguishes between targeted **Harassment** (including Cyberbullying & Hate Speech) and general **Offensive** language (vulgarity without harmful intent).

This repository hosts the dataset consolidation scripts, manual annotation guidelines, inter-annotator agreement protocols, and the training pipeline for our **XLM-RoBERTa Large** model.

## 👤 Author Information
* **Author:** Thilakasiri P.D.
* **Index Number:** E/20/397
* **Institution:** University of Peradeniya
* **Supervisor:** Dr. Eng. Sampath Deegalla

## 📌 Project Overview
* **Goal:** To create a robust detection system for low-resource languages and generate a massive, high-quality annotated corpus for the Sinhala NLP community.
* **Core Innovation:** We utilize a **3-Model Ensemble** (XLM-R, SinLlama, SinBERT) to "pseudo-label" a large unlabelled dataset (~65,000 documents), effectively solving the data scarcity problem in Sinhala NLP.

### Classification Classes (The "Umbrella" Approach)
We shifted our research direction to use **"Harassment"** as an umbrella term to capture broader forms of abuse.
1.  **Harassment (Cyberbullying & Hate Speech):** Targeted behavior meant to degrade, threaten, or intimidate. Includes threats of violence, self-harm encouragement, and attacks on family/ethnicity.
2.  **Offensive:** Content that violates social norms (profanity, crude jokes, "blue" humor) but lacks a specific target or malicious intent.
3.  **Normal:** Standard, respectful communication.

## 🧪 Methodology & Architecture

### 1. Data Curation (Human-in-the-Loop)
* **Consolidation:** Merged multiple raw datasets and removed duplicates.
* **Ground Truth:** Established via rigorous **Manual Annotation** and verified using **Inter-Annotator Agreement** (Peer Review) to resolve linguistic ambiguities in Singlish.
* **Preprocessing:** Utilized the **Google Transliteration API** to standardize Singlish text into Sinhala script.

### 2. Model Architecture: XLM-RoBERTa Large
We fine-tuned the **XLM-RoBERTa Large** model (550M parameters) with a custom classification head designed to prevent overfitting ("misfitting"):
* **Input:** 1024-dimensional latent vector from the `<s>` (CLS) token.
* **Dense Layer:** A fully connected linear layer with **Tanh activation** to extract non-linear semantic features.
* **Dropout Layer:** Implemented to randomly deactivate neurons ($p=0.1$) during training, forcing the model to learn robust linguistic patterns rather than memorizing specific words.

### 3. Ensemble Pseudo-Labeling (Final Phase)
We are currently fine-tuning **SinLlama-7B** and **SinBERT** to work alongside XLM-R. The consensus of these three models will be used to automatically label a massive raw dataset, creating the largest available corpus for Sinhala abusive language detection.

## 📊 Results (Preliminary)
We validated our model using **5-Fold Stratified Cross-Validation** to ensure stability across different data splits.

| Metric | Result |
| :--- | :--- |
| **Peak Accuracy (Fold 3)** | **80.49%** |
| **Average Accuracy** | **76.43%** |
| **Optimization** | Overfitting observed after Epoch 3; addressed via Early Stopping. |

## 📂 Repository Structure

```text
SinhSafe/
├── data/                       # Dataset directory
├── debug_results/              # Logs for debugging runs
├── models/                     # Saved model checkpoints
├── notebooks/                  # Jupyter notebooks for experiments
├── results/                    # Output from training folds
├── src/                        # Source code directory
├── venv/                       # Virtual environment
├── .gitignore                  # Git ignore file
├── calculate_f1.py             # Script to calculate F1 scores specifically
├── check_gpu.py                # Utility to verify RTX 3090 availability
├── data_process_output.log     # Logs from data preprocessing steps
├── debug_train.py              # Lightweight training script for debugging
├── f1_scores.log               # Log file specifically for F1 metrics
├── process_data.py             # Main script for data cleaning & consolidation
├── requirements.txt            # Project dependencies (PyTorch, Transformers)
├── test_model.py               # Script for inference on new text
├── train_cv.py                 # Main 5-Fold Cross-Validation training script
├── training_output.log         # General training logs
└── training_output_final_xlm.log # Logs for the final champion model run
