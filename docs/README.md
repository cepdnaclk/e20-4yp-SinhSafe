---
layout: home
permalink: index.html

# Please update this with your repository name and title
repository-name: e20-4yp-SinhSafe
title: A Benchmark & Dataset for Sinhala Cyberbullying Detection
---

[comment]: # "This is the standard layout for the project, but you can clean this and use your own template"

![SinhSafe Project Banner](./data/cover_page.jpg)

# SinhSafe: A Benchmark & Dataset for Sinhala Cyberbullying Detection

<div align="center">
    <img src="./data/thumbnail.png" width="200">
</div>

#### Team

- e20397, Thilakasiri P.D., [email](mailto:e20397@eng.pdn.ac.lk)

#### Supervisors

- Dr. Eng. Sampath Deegalla, [email](sampath@eng.pdn.ac.lk)

#### Table of content

1. [Abstract](#abstract)
2. [Related works](#related-works)
3. [Methodology](#methodology)
4. [Experiment Setup and Implementation](#experiment-setup-and-implementation)
5. [Results and Analysis](#results-and-analysis)
6. [Conclusion](#conclusion)
7. [Publications](#publications)
8. [Links](#links)

---

<!-- 
DELETE THIS SAMPLE before publishing to GitHub Pages !!!
This is a sample image, to show how to add images to your page. To learn more options, please refer [this](https://projects.ce.pdn.ac.lk/docs/faq/how-to-add-an-image/)
![Sample Image](./images/sample.png) 
-->


## Abstract

**SinhSafe** is an advanced cyberbullying detection framework specifically designed for the linguistic complexities of the Sinhala language and its code-mixed variant, Singlish. Addressing the critical gap in low-resource language safety tools, this project leverages state-of-the-art multilingual transformer models, specifically **XLM-RoBERTa (XLM-R)**, **SinBERT**, and **SinhLlama**, to classify social media content into three distinct categories: *Cyberbullying*, *Offensive*, and *Normal*.

A key innovation of SinhSafe is its robust preprocessing pipeline, which integrates a hybrid transliteration system combining Google API integration with a custom offline rule-based converter. This ensures high-accuracy normalization of Singlish text into Sinhala script before classification. By establishing a new benchmark and introducing a large pseudo-labeled dataset, SinhSafe aims to provide nuanced content moderation solutions for Sri Lankan social media platforms, promoting safer online communities through AI-driven intervention.

## Related works

Research into Sinhala cyberbullying detection has been constrained by a lack of diverse datasets and resources, a challenge highlighted by **Viswanath & Kumar (2023)**. 

**Datasets & Classification Taxonomy**
Most existing work, such as the *SOLD* and *semi-SOLD* datasets by **Ranasinghe et al. (2022)**, focuses on binary classification (Offensive vs. Not Offensive). We identified that binary labels are often insufficient for nuanced moderation. Consequently, our project draws significant inspiration from **Mathew et al. (2021)** and their *HateXplain* dataset. Following their approach, we moved beyond binary detection to a **3-class taxonomy (Cyberbullying, Offensive, Normal)**, allowing for more fine-grained analysis of harmful content.

**Model Architectures**
In the domain of model selection, **Dhananjaya et al. (2022)** demonstrated that monolingual models like **SinBERT** often outperform multilingual alternatives on small, specific Sinhala datasets. Conversely, the recent introduction of **SinLlama** by **Aravinda et al. (2025)** provides a foundation for testing Large Language Model (LLM) capabilities in Sinhala. SinhSafe benchmarks these distinct architectures (Multilingual XLM-R vs. Monolingual SinBERT vs. SinLlama) to determine the optimal approach.

**Preprocessing**
To handle the code-mixing prevalent in Sri Lankan social media, we leveraged insights from **SinLingua (Sameera et al., 2023)**, adopting their strategies for lemmatization and Singlish-to-Sinhala conversion within our hybrid preprocessing pipeline.

## Methodology

The SinhSafe framework consists of three main stages:

1.  **Data Collection & Pseudo-Labeling:** Aggregating a large dataset of Sinhala/Singlish comments and using semi-supervised learning to generate fine-grained labels (Normal, Offensive, Cyberbullying).
2.  **Hybrid Preprocessing Pipeline:**
    * **Layer 1 (Google API):** High-accuracy online transliteration for Singlish content.
    * **Layer 2 (Offline Backup):** A rule-based dictionary and phoneme mapper to handle transliteration when offline.
    * **Layer 3 (Regex Cleaning):** Removal of URLs, usernames, and noise.
3.  **Model Architecture:** Fine-tuning transformer-based models (**XLM-R**, **SinBERT**, **SinhLlama**) on the processed dataset to learn context-aware sentiment representations.

## Experiment Setup and Implementation

The models were trained using the Hugging Face `transformers` library on an NVIDIA RTX 3090 GPU. The dataset was split into training, validation, and testing sets. We employed techniques such as:
* **Tokenizer:** XLM-RoBERTa tokenizer (SentencePiece) for handling multilingual vocabulary.
* **Optimization:** AdamW optimizer with a learning rate scheduler.
* **Evaluation Metrics:** Macro F1-Score, Weighted F1-Score, Precision, and Recall.

## Results and Analysis

We are currently benchmarking three models: **XLM-RoBERTa**, **SinBERT**, and **SinhLlama**. 

**Phase 1 Results: XLM-RoBERTa (Large)**
Initial experiments with the multilingual XLM-R model have demonstrated strong performance on the code-mixed dataset:

* **Overall Performance:** Achieved a **Macro F1-Score of 0.82**.
* **Safety Profile:** The model maintained a **0.93 F1-Score for the 'Normal' class**, ensuring minimal false positives.
* **Detection Capability:** Despite the complexity of code-mixed data, the system successfully identifies subtle offensive content.

**Phase 2: Comparative Benchmarking (In Progress)**
Experiments are currently underway for **SinBERT** and **SinhLlama**. Once completed, their performance will be tabulated here to determine the most effective architecture for Sinhala cyberbullying detection.

## Conclusion

SinhSafe successfully establishes a robust pipeline for Sinhala cyberbullying detection, combining a novel hybrid transliteration system with powerful transformer models. Preliminary results with XLM-RoBERTa indicate that fine-grained classification in low-resource settings is highly achievable with an F1-score of 0.82. The ongoing comparison with SinBERT and SinhLlama will provide a definitive benchmark for the research community, identifying the optimal model architecture for protecting the Sri Lankan digital ecosystem.

## Publications
[//]: # "Note: Uncomment each once you uploaded the files to the repository"

<!-- 1. [Semester 7 report](./) -->
<!-- 2. [Semester 7 slides](./) -->
<!-- 3. [Semester 8 report](./) -->
<!-- 4. [Semester 8 slides](./) -->
<!-- 5. Author 1, Author 2 and Author 3 "Research paper title" (2021). [PDF](./). -->


## Links

[//]: # ( NOTE: EDIT THIS LINKS WITH YOUR REPO DETAILS )

- [Project Repository](https://github.com/cepdnaclk/e20-4yp-SinhSafe)
- [Project Page](https://cepdnaclk.github.io/e20-4yp-SinhSafe)
- [Department of Computer Engineering](http://www.ce.pdn.ac.lk/)
- [University of Peradeniya](https://eng.pdn.ac.lk/)

[//]: # "Please refer this to learn more about Markdown syntax"
[//]: # "https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet"
