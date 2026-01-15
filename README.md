# SinhSafe: Sinhala Cyberbullying Detection System

### 🌐 [Click Here to Visit the Project Website](https://cepdnaclk.github.io/e20-4yp-SinhSafe)

---

## 👥 Project Team
| Role | Name | E-Number | Email |
| :--- | :--- | :--- | :--- |
| **Author** | **Thilakasiri P.D.** | E/20/397 | [e20397@eng.pdn.ac.lk](mailto:e20397@eng.pdn.ac.lk) |
| **Supervisor** | **Dr. Eng. Sampath Deegalla** | - | [sampath@eng.pdn.ac.lk](mailto:sampath@eng.pdn.ac.lk) |

---

## 🚀 About the Project
**SinhSafe** is a deep learning framework designed to detect cyberbullying in Sinhala and Singlish (code-mixed) social media comments. It classifies text into three categories:
1.  **Normal**
2.  **Offensive**
3.  **Cyberbullying**

This repository contains the source code for the hybrid preprocessing pipeline, the model training scripts (XLM-R, SinBERT), and the dataset processing utilities.

## 📂 Repository Structure

```text
├── data/                   # Dataset files (Excel/CSV)
│   ├── processed_ground_truth/  # Cleaned files ready for training
│   └── ...
├── models/                 # Saved model weights (Ignored by Git if large)
├── src/                    # Helper scripts for preprocessing
│   ├── process_ground_truth.py
│   └── ...
├── docs/                   # Website source code (Do not edit unless changing the site)
├── process_data.py         # Main script to run the pipeline
├── offline_transliteration.py # Backup transliteration tool
└── README.md               # This file
