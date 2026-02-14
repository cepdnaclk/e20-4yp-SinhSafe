# SinhSafe: Fine-Grained Cyberbullying Detection for Sinhala

**SinhSafe** is a research project focused on detecting cyberbullying in the Sinhala language. Unlike binary classification systems (Bully/Not Bully), this project aims to perform **fine-grained classification** to distinguish between targeted harassment, general offensive language, and non-harmful content.

This repository hosts the dataset preprocessing pipeline, model training scripts, and comparative analysis between Monolingual and Cross-lingual architectures.

## 📌 Project Overview

* **Project Type:** Final Year Project (Computer Engineering)
* **Institution:** University of Peradeniya
* **Supervisor:** Dr. Eng. Sampath Deegalla
* **Goal:** To compare the efficacy of Monolingual models against Cross-lingual models in low-resource settings.

### Classification Classes
The model classifies text into three distinct categories:
1.  **Targeted Cyberbullying:** Content specifically aimed at an individual/group with malicious intent.
2.  **General Offensive:** Profanity or toxicity not directed at a specific target (e.g., conversational swearing).
3.  **Non-Harmful:** Neutral or positive content.

## 🧪 Methodology

We are benchmarking two approaches to determine the state-of-the-art for Sinhala NLP in this domain:

| Approach | Models Tested |
| :--- | :--- |
| **Monolingual** | **SinBERT**, **SinLlama** (Fine-tuned on Sinhala-only corpora) |
| **Cross-Lingual** | **XLM-R** (Leveraging transfer learning from high-resource languages) |

## 📂 Repository Structure

```text
SinhSafe/
├── data/                  # (Ignored by Git)
│   ├── raw/               # Original raw dataset
│   ├── processed/         # Cleaned/Transliterated data ready for training
│   └── to_label/          # Unlabeled data batches
├── models/                # Saved model checkpoints (Ignored by Git)
├── notebooks/             # Jupyter notebooks for EDA and experiments
│   ├── 01_EDA.ipynb
│   └── 02_Preprocessing_Pipeline.ipynb
├── src/                   # Source code for the project
│   ├── preprocessing.py   # Text cleaning and tokenization pipeline
│   └── utils.py           # Helper functions
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation