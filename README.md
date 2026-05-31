# Automated Sensitivity Classification of Arabic Documents

> Automated classification of Arabic documents by sensitivity level using NLP and machine learning, aligned with SDAIA data governance regulations.


##  Project Overview

This project compares three Arabic NLP models for automatically classifying Arabic administrative documents into sensitivity levels based on **NDMO/SDAIA** regulations:

| Level      | Arabic      | Description |
|------------|-------------|-------------|
| Public     | عام         | Openly shareable |
| Restricted | مقيد        | Internal use only |
| Secret     | سري         | Confidential |
| Top Secret | سري للغاية | Highly sensitive |

## Models Compared

| Model            | Type                 | Approach |
|------------------|----------------------|----------|
| **AraBERT**      | Encoder (BERT)       | Fine-tuned classification |
| **LLaMA 3.1-8B** | Decoder (LLM)        | Fine-tuned + Zero/One/Three-Shot |
| **ALLaM-7B**     | Decoder (Arabic LLM) | Fine-tuned + Zero/One/Three-Shot |


##  Repository Structure

```
Automated-Sensitivity-Classification-of-Arabic-Documents/
│
├── README.md
├── requirements.txt
│
├── arabert/
│   └── AraBERT.ipynb               # AraBERT fine-tuning & evaluation
│
├── llama/
│   ├── LLaMA_Model.ipynb           # LLaMA fine-tuning & evaluation
│   └── LLaMA_MultiShots.ipynb      # LLaMA zero/one/three-shot evaluation
│
└── alam/
    ├── ALLaM_Finetuned.ipynb       # ALLaM fine-tuning & evaluation
    └── ALLaM_MultiShots.ipynb      # ALLaM zero/one/three-shot evaluation
```


##  Dataset

- **File:** `synthetic_sensitivity_dataset_v3.csv`
- **Type:** Synthetic Arabic administrative documents
- **Size:** ~40,000 documents
- **Columns:** `content`, `sensitivity_level`
- **Split:** 70% train / 15% validation / 15% test (seed=42)

>  Due to file size, the dataset is stored on Google Drive.
>  Download: https://drive.google.com/file/d/15Zg_ZOWWd596hDz-cW5luUYPaSjvxpyk/view?usp=sharing


##  Setup & Requirements

### 1. Clone the Repository

```bash
git clone https://github.com/Taghreed-Alzahrani/Automated-Sensitivity-Classification-of-Arabic-Documents.git
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Google Colab (Recommended)

All notebooks are designed to run on **Google Colab** with GPU support (L4 or T4).

Steps:
1. Upload the notebook to Colab
2. Mount Google Drive and download the dataset from the link above
3. Add your Hugging Face token to Colab Secrets as `HF_TOKEN`
4. Run cells in order

>  LLaMA and ALLaM require a Hugging Face account with access to gated models.


##  Running the Notebooks

| Notebook | What it does |
|----------|-------------|
| `arabert/AraBERT.ipynb` | Fine-tunes AraBERT and evaluates on test set |
| `llama/LLaMA_Model.ipynb` | Fine-tunes LLaMA 3.1-8B with LoRA and evaluates |
| `llama/LLaMA_MultiShots.ipynb` | Runs zero/one/three-shot on base LLaMA |
| `alam/ALLaM_Finetuned.ipynb` | Fine-tunes ALLaM-7B with QLoRA and evaluates |
| `alam/ALLaM_MultiShots.ipynb` | Runs zero/one/three-shot on base ALLaM |


##  Results Summary

| Model        | Setting    | Accuracy | Macro F1 |
|--------------|------------|----------|----------|
| AraBERT      | Fine-tuned |   0.75   |   0.76   |
| LLaMA 3.1-8B | Fine-tuned |   0.72   |   0.73   |
| LLaMA 3.1-8B | Zero-Shot  |   0.24   |   0.10   |
| LLaMA 3.1-8B | One-Shot   |   0.24   |   0.10   |
| LLaMA 3.1-8B | Three-Shot |   0.24   |   0.10   |
| ALLaM-7B     | Fine-tuned |   0.73   |   0.73   |
| ALLaM-7B     | Zero-Shot  |   0.25   |   0.14   |
| ALLaM-7B     | One-Shot   |   0.29   |   0.18   |
| ALLaM-7B     | Three-Shot |   0.32   |   0.22   |


##  Tech Stack

- **Python** 3.10+
- **PyTorch** + **Transformers** (Hugging Face)
- **PEFT / LoRA** for parameter-efficient fine-tuning
- **TRL** (SFTTrainer) for instruction fine-tuning
- **AraBERT** (`aubmindlab/bert-base-arabertv02`)
- **LLaMA 3.1-8B-Instruct** (`meta-llama/Meta-Llama-3.1-8B-Instruct`)
- **ALLaM-7B-Instruct** (`humain-ai/ALLaM-7B-Instruct-preview`)
- **Google Colab** (GPU: L4 / T4)


##  Compliance

This project aligns with:
- **NDMO** — National Data Management Office classification standards
- **SDAIA** — Saudi Data & AI Authority data governance framework


##  Authors

Reem Ghazi Alosaimi
Mashael Salman Abdali
Taghreed Saleh Alzahrani
Nora Targ Alshibi
Supervised by: Dr. Omaima Fallatah
Graduation Project, Data Science Department, College of Computing, Umm Al-Qura University


##  License
This project is submitted as a graduation project and is intended for academic use only.