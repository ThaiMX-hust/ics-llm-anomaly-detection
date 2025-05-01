# 🛡️ Enhancing ICS Intrusion Detection using Large Language Models & Machine Learning

This repository implements anomaly detection for Industrial Control Systems (ICS) using **classical machine-learning (ML)**, **deep-learning (DL)**, and **transformer-based large language models (LLMs)** such as BERT, DistilBERT, and SecBERT.

---

## 📁 Project Structure


```text
.
├── main_train.py            # Train DL models (AE, LSTM, CNN, …)
├── main_model_tuning.py     # Hyperparameter search & evaluation
├── classical_ml.py          # Train classical ML models (SVM, RF, AdaBoost, k-NN)
├── llm_models.py            # Transformer models: BERT, DistilBERT, SecBERT
├── data_loader.py           # Load & preprocess datasets
├── utils.py                 # Helper functions, metrics, splits, …
│
├── models/                  # Saved configs (.json) — NO large weights committed
├── data/                    # Place SWAT / WADI / BATADAL CSVs here (ignored)
├── outputs/                 # Metrics (.npy), logs, detection curves
├── plots/                   # Generated figures / PDFs
├── references/              # Papers (e.g. ESORICS 2022)
│   └── esorics2022-ics-anomaly-detection.pdf
│
├── requirements.txt
├── .gitignore
└── README.md                # This file
```
## 📊 Models Supported
Category	Algorithms
Classical ML	SVM • Random Forest • AdaBoost • k-Nearest-Neighbors
Deep Learning	Autoencoder • CNN • LSTM • GRU • DNN • Linear
Transformer / LLM	BERT • DistilBERT • SecBERT

Classical models operate on engineered statistical features; DL & LLM models forecast the next sensor state from historical windows.

## 📦 Datasets
Dataset	Domain	Size	Notes
BATADAL	Water-distribution simulation	58 k rows	Download TAR archive

Obtain raw CSV files from the official sources.

Place them under data/<DATASET_NAME>/... (same filenames).

data_loader.py assumes the original dataset structure.

## 🚀 Getting Started
1 – Clone & install
bash
Copy
Edit
<code>
git clone git@github.com:<your-username>/ics-llm-anomaly-detection.git
cd ics-llm-anomaly-detection
python -m venv .venv && source .venv/bin/activate   # optional virtual-env
pip install -r requirements.txt
<code>
2 – Train a Deep-Learning model (LSTM × SWAT)
bash
Copy
Edit
python main_train.py \
  --model LSTM \
  --run_name lstm
3 – Train a Classical ML model (SVM × SWAT)
bash
Copy
Edit
python classical_ml.py \
  --model SVM \
  --run_name svm
4 – Fine-tune an LLM (SecBERT × WADI) *
bash
Copy
Edit
python llm_models.py 

* Requires GPU + transformers ≥ 4.


## 📈 Metrics
Point-F1 (per timestep)


Accuracy • Precision • Recall

Detection latency

False-positive rate (FPR)

NA-early (Numenta-style early detection)


## 📜 License
Distributed under the Apache License 2.0 – see LICENSE.

## 🙏 Acknowledgments
Baseline framework from “ESORICS 2022 – Reconstruction-based Anomaly Detection in ICS” by Lujo Bauer & Clement Fung

Classical ML pipelines powered by scikit-learn

Transformer back-ends by HuggingFace Transformers

Built with ❤️ using PyTorch, TensorFlow/Keras, and scikit-learn.
