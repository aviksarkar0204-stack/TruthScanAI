# FakeGuard 🔍

> End-to-end fake news detection project — from raw ISOT dataset to deployed web application. Every component built independently: data engineering, NLP preprocessing pipeline, classical ML training, PyTorch LSTM with word embeddings, Flask API, and custom UI.

---

## Overview

FakeGuard is a fake news detector that classifies news articles as **Real** or **Fake**. Trained on 44,898 articles from the ISOT Fake News Dataset, the project includes two approaches — classical ML (SVM + Logistic Regression) and a deep learning upgrade using a PyTorch LSTM with learned word embeddings. The deployed app supports SVM and Logistic Regression with real-time confidence scores, a credibility circle, and a fully custom dark/light theme UI.

---

## Demo

[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Spaces-yellow)](https://huggingface.co/spaces/Avik128/TruthScanAI)

> Paste any news article → select a model → get instant prediction with confidence scores.

---

## Built With

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-154F5B?style=for-the-badge&logo=python&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![HTML5](https://img.shields.io/badge/HTML5-E34F26?style=for-the-badge&logo=html5&logoColor=white)
![CSS3](https://img.shields.io/badge/CSS3-1572B6?style=for-the-badge&logo=css3&logoColor=white)
![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Git](https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=git&logoColor=white)
![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)

---

## Final Model Comparison

| Model | Accuracy | Approach |
|---|---|---|
| Naive Bayes | 93.43% | Classical ML |
| Logistic Regression | 98.72% | Classical ML |
| Random Forest | 99.11% | Classical ML |
| SVM (LinearSVC) | 99.51% | Classical ML |
| **LSTM + Embeddings** | **99.81%** ✅ | **Deep Learning** |

LSTM with learned word embeddings outperforms all classical ML models.

---

## Features

- Combined two separate CSV files (Real + Fake) into one clean dataset
- Engineered a `content` column by combining `title` + `text` for richer features
- Custom 7-step NLP preprocessing pipeline built from scratch
- Classical ML model comparison across 4 algorithms
- PyTorch LSTM with learned word embeddings (deep learning upgrade)
- Real-time confidence scores (Real/Fake probability bars)
- Credibility circle — animated score indicator
- Model switching — user can choose between SVM and Logistic Regression
- Dark/Light theme toggle
- Flask REST API backend
- Custom HTML/CSS/JavaScript frontend

---

## Project Structure

```
FakeGuard/
│
├── app.py                         ← Flask backend
├── requirements.txt
│
├── models/
│   ├── svm.pkl                    ← trained SVM pipeline
│   ├── logistic_regression.pkl    ← trained LR pipeline
│   └── fakeguard_lstm.pth         ← trained LSTM weights
│
├── src/
│   ├── __init__.py
│   └── preprocessing.py           ← custom NLP preprocessing
│
├── templates/
│   └── index.html                 ← custom UI
│
└── notebooks/
    ├── main.ipynb                 ← classical ML training notebook
    └── LSTM_Embedding.ipynb       ← PyTorch LSTM training notebook
```

---

## Dataset

**ISOT Fake News Dataset** (Kaggle)

| File | Content | Count |
|---|---|---|
| `True.csv` | Real news from Reuters | 21,417 articles |
| `Fake.csv` | Fake news from unreliable sources | 23,481 articles |
| **Combined** | **Shuffled and merged** | **44,898 articles** |

### Data Engineering Steps

**1. Add labels before combining:**
```python
df_true['label'] = 1   # 1 = Real
df_fake['label'] = 0   # 0 = Fake
```

**2. Combine and shuffle:**
```python
df = pd.concat([df_true, df_fake], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
```

**3. Drop irrelevant columns:**
```python
df = df.drop(columns=['subject', 'date'])
```

`subject` was dropped to prevent **data leakage** — the subject column correlates strongly with the label which would give the model an unfair shortcut instead of learning from actual writing patterns.

**4. Combine title + text into content:**
```python
df['content'] = df['title'] + ' ' + df['text']
```

---

## Approach 1 — Classical ML (SVM + TF-IDF)

### NLP Preprocessing Pipeline

```
Raw Text → Remove HTML → Lowercase → Remove Punctuation
        → Tokenize → Remove Stopwords → Stem → Join
```

### Model Training

| Model | Accuracy | Training Time |
|---|---|---|
| Naive Bayes | 93.43% | 5.24s |
| Logistic Regression | 98.72% | 5.66s |
| Random Forest | 99.11% | 88.94s |
| **SVM (LinearSVC)** | **99.51%** | **5.60s** |

SVM selected as primary deployed model — highest accuracy and fast training.

---

## Approach 2 — Deep Learning (PyTorch LSTM + Word Embeddings)

### Why a separate approach?

TF-IDF treats each word independently with no sense of order or meaning. LSTM + Embeddings learns:
- **Word representations** — semantically similar words get similar vectors
- **Sequential patterns** — word order and context matter
- **Writing style** — how words flow together, not just which words appear

### Preprocessing (lighter than classical ML)

Stemming is harmful for embeddings — embeddings work better with real words, not stemmed forms. So the DL pipeline uses:

```
Raw Text → Lowercase → Remove HTML → Remove Punctuation → Collapse spaces
```

No stopword removal, no stemming.

### Vocabulary Building

```python
vocab_size = 20000
vocab = ['<PAD>', '<UNK>'] + [word for word, count in word_counts.most_common(vocab_size)]
word2idx = {word: idx for idx, word in enumerate(vocab)}
```

- Top 20,000 most frequent words kept
- `<PAD>` (index 0) — pads shorter articles to uniform length
- `<UNK>` (index 1) — replaces words outside the vocabulary

### Sequence Encoding + Padding

Every article is converted to a sequence of integers, then padded/truncated to `max_len = 500`:

```
"trump said president" → [2, 45, 4, 0, 0, 0, ...]  (padded to 500)
```

### LSTM Architecture

```
Input (batch, 500)          ← integer sequences
    ↓
nn.Embedding(20002, 64)     ← lookup table: integer → 64-dim vector
    ↓
nn.LSTM(64, 128)            ← reads sequence, hidden_size=128
    ↓
Last timestep output        ← summary of full article
    ↓
nn.Dropout(0.3)             ← regularization
    ↓
nn.Linear(128, 1)           ← classification head
    ↓
nn.Sigmoid()                ← probability 0-1
    ↓
Output (Real/Fake)
```

### Training

- **Loss**: `nn.BCELoss()` — binary cross entropy
- **Optimizer**: `Adam (lr=0.001)` — initialized AFTER `model.to(device)`
- **Epochs**: 50
- **Batch size**: 64
- **Hardware**: Google Colab T4 GPU

### Key Lesson — Optimizer Bug

A critical PyTorch mistake was caught and fixed during training:

```python
# WRONG — optimizer holds CPU parameter references
model = FakeGuardLSTM(...)
optimizer = optim.Adam(model.parameters())  # CPU params
model = model.to(device)                    # model moves to GPU, optimizer doesn't

# CORRECT — always move model to device before creating optimizer
model = FakeGuardLSTM(...)
model = model.to(device)                    # GPU first
optimizer = optim.Adam(model.parameters())  # GPU params ✅
```

This bug caused the model to predict randomly (loss stuck at 0.693, accuracy 39%) despite the GPU being active. Fixing it unlocked full convergence.

### Results

```
Epoch 20/50 | Avg Loss: 0.0100
Epoch 30/50 | Avg Loss: 0.0050
Epoch 40/50 | Avg Loss: 0.0030
Epoch 50/50 | Avg Loss: 0.0010

Accuracy: 99.81% ✅
```

Loss curve drops cleanly from 0.65 → near 0 by epoch 20 and stays flat — perfect convergence.

---

## Flask API

### Routes

| Route | Method | Description |
|---|---|---|
| `/` | GET | Serves the HTML frontend |
| `/predict` | POST | Accepts article text, returns verdict |

### Request Format

```json
{
    "article": "Russia launched missile strikes on Ukrainian cities...",
    "model": "svm"
}
```

### Response Format

```json
{
    "verdict": "real",
    "real_prob": 91,
    "fake_prob": 9
}
```

---

## Installation & Running Locally

```bash
git clone https://github.com/aviksarkar0204-stack/FakeGuard.git
cd FakeGuard
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000` in your browser.

---

## What I Learned

- Identifying and preventing data leakage from feature columns
- Why shuffling matters before train/test split
- TF-IDF capturing writing style differences between real and fake news
- Building vocabulary from scratch for NLP with PyTorch
- Word embeddings — learned dense representations vs sparse TF-IDF vectors
- PyTorch LSTM for text classification
- Critical PyTorch bug — optimizer must be created after `model.to(device)`
- GPU training on Google Colab T4

---

## Author

**Avik Sarkar**
- GitHub: [@aviksarkar0204-stack](https://github.com/aviksarkar0204-stack)
- Hugging Face: [Avik128](https://huggingface.co/Avik128)

---

## License

MIT License — free to use and modify.
