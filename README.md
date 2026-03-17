# FakeGuard 🔍

> End-to-end fake news detection project — from raw ISOT dataset to deployed web application. Every component built independently: data engineering, NLP preprocessing pipeline, model training, Flask API, and custom UI.

---

## Overview

FakeGuard is a fake news detector that classifies news articles as **Real** or **Fake**. Trained on 44,898 articles from the ISOT Fake News Dataset, the app supports two ML models — SVM and Logistic Regression — with real-time confidence scores, a credibility circle, and a fully custom dark/light theme UI.

---

## Demo

> Paste any news article → select a model → get instant verdict with confidence scores and key indicators.

**Supported Models:**
| Model | Accuracy |
|---|---|
| SVM (LinearSVC) | 99.51% |
| Logistic Regression | 98.72% |

---

## Features

- Combined two separate CSV files (Real + Fake) into one clean dataset
- Engineered a `content` column by combining `title` + `text` for richer features
- Custom 7-step NLP preprocessing pipeline built from scratch
- Model comparison across 4 algorithms
- Real-time confidence scores (Real/Fake probability bars)
- Credibility circle — animated score indicator
- Model switching — user can choose between SVM and Logistic Regression
- Key indicators section — changes based on verdict
- Dark/Light theme toggle
- Flask REST API backend
- Custom HTML/CSS/JavaScript frontend

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3 |
| Backend | Flask |
| ML | scikit-learn |
| NLP | NLTK |
| Frontend | HTML, CSS, JavaScript, Tailwind CSS |
| Data | pandas, numpy |
| Visualization | matplotlib, seaborn |
| Model Saving | joblib |

---

## Project Structure

```
FakeGuard/
│
├── app.py                        ← Flask backend
├── requirements.txt
│
├── models/
│   ├── svm.pkl                   ← trained SVM pipeline
│   └── logistic_regression.pkl   ← trained LR pipeline
│
├── src/
│   ├── __init__.py
│   └── preprocessing.py          ← custom NLP preprocessing
│
├── templates/
│   └── index.html                ← custom UI
│
└── notebooks/
    └── main.ipynb                ← training notebook
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

Shuffling prevents ordering bias — without it, all real news would appear first and all fake news last, causing issues during train/test split.

**3. Drop irrelevant columns:**
```python
df = df.drop(columns=['subject', 'date'])
```

`subject` was dropped to prevent **data leakage** — the subject column correlates strongly with the label (all fake news had specific subject tags) which would give the model an unfair shortcut instead of learning from actual writing patterns.

**4. Combine title + text into content:**
```python
df['content'] = df['title'] + ' ' + df['text']
```

Combining both columns gives the model richer signal — the headline style AND the article body writing style together.

---

## NLP Preprocessing Pipeline

Every preprocessing function was written from scratch. The pipeline runs in this order:

```
Raw Article Text
      ↓
1. Remove HTML tags      → strips <br />, <p>, etc.
      ↓
2. Lowercase             → converts all text to lowercase
      ↓
3. Remove punctuation    → keeps only letters, replaces rest with space
      ↓
4. Tokenization          → splits text into individual words
      ↓
5. Remove stopwords      → removes common words (the, is, a, etc.)
      ↓
6. Stemming              → reduces words to root form (running → run)
      ↓
7. Join tokens           → converts token list back to string
      ↓
Clean Text → TF-IDF Vectorizer → Model
```

### Example

```
Input:  "SHOCKING: Russia SECRETLY surrenders! 
         Sources who CANNOT be named have confirmed..."

Output: "shock russia secretli surrend sourc cannot name confirm"
```

### Why these steps matter for fake news

| Step | Why it helps |
|---|---|
| Lowercase | "BREAKING" and "breaking" treated as same word |
| Remove punctuation | Fake news uses excessive "!!!" — removes noise |
| Stopwords | Removes "the", "is" — keeps content words |
| Stemming | "shocking", "shocked", "shocks" → same root |

---

## Model Training

### Feature Extraction
- **TF-IDF Vectorizer** — converts cleaned text to numerical features
- Learns vocabulary from training data only (no data leakage)
- Fake and real news have very distinct vocabulary — TF-IDF captures this perfectly

### Model Comparison

| Model | Accuracy | Training Time |
|---|---|---|
| Naive Bayes | 93.43% | 5.24s |
| Logistic Regression | 98.72% | 5.66s |
| Random Forest | 99.11% | 88.94s |
| **SVM (LinearSVC)** | **99.51%** | **5.60s** |

SVM was selected as the primary model — highest accuracy AND fast training time. Random Forest achieved similar accuracy but took 16x longer to train.

### Why such high accuracy?

Fake and real news have extremely distinct writing patterns that TF-IDF picks up automatically:

| Real News | Fake News |
|---|---|
| Formal, neutral tone | Sensationalist language |
| Cited sources ("Reuters", "AP") | Vague unnamed sources |
| Specific facts and dates | Emotional manipulation |
| Professional vocabulary | "SHOCKING", "EXPOSED", "BREAKING" |
| Complete sentences | Excessive punctuation!!! |

### sklearn Pipeline

Each model is saved as a full sklearn Pipeline:

```python
Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LinearSVC())
])
```

This ensures TF-IDF and model are bundled together — clean inference with no data leakage.

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

### Confidence Score Logic

- **SVM** — uses `decision_function()` score converted to 0-100 range
- **Logistic Regression** — uses `predict_proba()` for true probabilities

---

## Installation & Running Locally

### 1. Clone the repository

```bash
git clone https://github.com/aviksarkar0204-stack/FakeGuard.git
cd FakeGuard
```

### 2. Create and activate conda environment

```bash
conda create -n fakeguard python=3.10
conda activate fakeguard
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
python app.py
```

### 5. Open in browser

```
http://127.0.0.1:5000
```

---

## Requirements

```
flask
scikit-learn
nltk
pandas
numpy
joblib
```

### NLTK Downloads

On first run, make sure these are downloaded:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')
```

---

## How It Works — End to End

```
User pastes news article in browser
            ↓
JavaScript sends POST request to /predict
            ↓
Flask receives article + model choice
            ↓
preprocessing() cleans the raw text
            ↓
sklearn Pipeline:
  TF-IDF vectorizes cleaned text
  Model predicts 0 (fake) or 1 (real)
            ↓
Confidence score calculated
            ↓
JSON response sent back to browser
            ↓
JavaScript updates UI:
  Verdict badge (✅ Real / 🚨 Fake)
  Credibility circle animates
  Confidence bars update
  Key indicators change
```

---

## Sample Predictions

**Real News Article:**
```
"Russia launched missile strikes on several Ukrainian cities overnight, 
targeting energy infrastructure according to Ukrainian officials. 
NATO Secretary General expressed concern over the escalating attacks."

→ ✅ Real News | Confidence: 91%
```

**Fake News Article:**
```
"SHOCKING: Russia SECRETLY surrenders to Ukraine! Putin EXPOSED! 
Sources who CANNOT be named confirmed this explosive revelation. 
Share before it gets deleted!"

→ 🚨 Fake News | Confidence: 96%
```

---

## Key Design Decisions

**Why drop the `subject` column?**
The subject column was highly correlated with labels — all fake news had specific subject tags. Using it would cause data leakage, letting the model cheat rather than learning actual writing patterns.

**Why combine `title` + `text`?**
Headlines carry strong signal (fake news headlines are clickbait-y) and article bodies carry writing style signal. Combining both gives maximum information to the model.

**Why shuffle after concat?**
Without shuffling, all real news (21,417 rows) appears first and all fake news (23,481 rows) last. Train/test split would then have imbalanced class distribution.

---

## What I Learned

- Combining multiple datasets correctly with proper label assignment
- Identifying and preventing data leakage from feature columns
- Feature engineering — combining columns for richer signal
- Why shuffling matters before train/test split
- TF-IDF capturing writing style differences between real and fake news
- Building and deploying Flask REST APIs
- Handling confidence scores for both SVM and probabilistic models

---

## Author

**Avik Sarkar**
- GitHub: [@aviksarkar0204-stack](https://github.com/aviksarkar0204-stack)
- Hugging Face: [Avik128](https://huggingface.co/Avik128)

---

## License

MIT License — free to use and modify.
