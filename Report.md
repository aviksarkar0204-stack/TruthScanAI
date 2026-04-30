# FakeGuard — Model Evaluation Report

> A critical analysis of three fake news detection models — SVM, Logistic Regression, and LSTM with Word Embeddings — comparing benchmark accuracy against real-world generalization.

---

## Overview

This report documents the evaluation of three models trained on the **ISOT Fake News Dataset** (44,898 articles). All three models achieve high benchmark accuracy, but real-world testing reveals important differences in generalization that benchmark numbers alone cannot capture.

---

## Benchmark Results (ISOT Test Set)

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Naive Bayes | 93.43% | — | — | — |
| Logistic Regression | 98.72% | — | — | — |
| Random Forest | 99.11% | — | — | — |
| SVM (LinearSVC) | 99.51% | — | — | — |
| **LSTM + Embeddings** | **99.81%** | — | — | — |

By benchmark numbers alone, LSTM is the clear winner.

---

## Real-World Test — BBC News Article

To evaluate generalization beyond the benchmark, all three models were tested on a **real BBC News article** that was never part of the training or test set.

### Article Tested

> **'Attacked 28 times in a day' — BBC visits heavily targeted US-UK base in Iraq**
>
> The BBC has been given access to a military base in Iraq where UK forces have been working together with their US counterparts during the conflict in the region. The US announced an extended but fragile ceasefire on the US-Israel war in Iran — but prior to the ceasefire up to 28 drones were fired at the base on a daily basis. "You hear weapons of destruction going off around you, and it's bloody difficult," an RAF air specialist at the base told the BBC's defence correspondent Jonathan Beale...

**Ground truth: Real News** (published by BBC, a verified news source)

### Results

| Model | Verdict | Real % | Fake % | Correct? |
|---|---|---|---|---|
| SVM (LinearSVC) | 🚨 Fake | 40% | **60%** | ❌ |
| Logistic Regression | 🚨 Fake | 33% | **67%** | ❌ |
| LSTM + Embeddings | 🚨 Fake | 0% | **100%** | ❌ |

All three models incorrectly classified this legitimate BBC article as fake news.

---

## Analysis — Why Did All Three Fail?

### The Training Data Problem

The ISOT dataset has a fundamental structural bias:

| Split | Source | Writing Style |
|---|---|---|
| Real News | Reuters only | Formal, neutral, wire service style |
| Fake News | Unreliable websites | Sensationalist, emotional, clickbait |

All three models learned to associate **Reuters writing style = Real** and **sensationalist language = Fake**. They never encountered legitimate news written in non-Reuters styles during training.

### Why the BBC Article Was Flagged

The BBC article contains several patterns the models learned to associate with fake news:

| Feature | Example | Why It Triggered |
|---|---|---|
| Dramatic quoted headline | `'Attacked 28 times in a day'` | Looks like clickbait to the model |
| Informal quotes | `"it's bloody difficult"` | Reuters never uses casual language |
| Opinionated language | `"Sharp differences have strained relations"` | Sounds editorial, not neutral |
| First-person narrative | Reporter's perspective included | Reuters uses third-person wire format only |

### Why LSTM Failed Worst

Despite having the highest benchmark accuracy (99.81%), LSTM generalized the worst — flagging the article as **100% fake** with zero uncertainty.

LSTM learned Reuters writing patterns more rigidly than classical ML models. Word embeddings encode sequential writing style, which means the model became extremely sensitive to how sentences flow — a style unique to Reuters. Any deviation from that style triggers a strong fake signal.

SVM and LR using TF-IDF are more forgiving — they look at word frequency rather than sequential patterns, so they're less sensitive to writing style differences.

---

## Key Insight — Accuracy vs Generalization

This experiment demonstrates one of the most important lessons in applied ML:

> **High benchmark accuracy does not guarantee real-world performance.**

| Metric | What It Measures |
|---|---|
| Benchmark Accuracy | How well the model memorized the training distribution |
| Generalization | How well the model handles unseen real-world data |
| Robustness | How well the model handles distribution shift and edge cases |

In production ML, **robustness matters more than accuracy**. A model that scores 99.81% on a clean benchmark but fails on real BBC articles is not production-ready.

### The Paradox

| Model | Benchmark Accuracy | Real-World Performance |
|---|---|---|
| SVM | 99.51% | Best (60% fake — least wrong) |
| LR | 98.72% | Middle (67% fake) |
| LSTM | **99.81%** | **Worst (100% fake)** |

The model with the highest accuracy performed worst on real-world data. This is called **overfitting to the training distribution**.

---

## Known Limitations

**1. Single-source real news**
All real news in ISOT comes from Reuters. The model has never seen legitimate news from BBC, Al Jazeera, The Hindu, Guardian, or any other outlet.

**2. No writing style diversity**
Reuters has a very specific wire service style. Any deviation — including legitimate journalistic styles — is treated as suspicious.

**3. Headline sensitivity**
Clickbait headlines are common in legitimate media today. The model incorrectly penalizes dramatic headlines even when the article body is factual.

**4. No contextual understanding**
TF-IDF and basic LSTM cannot understand context. "Attacked 28 times" is dramatic but factual — the model has no way to distinguish intent from language style.

---

## Planned Improvements

### Short Term — Diverse Training Data
Retrain on **WELFake Dataset** (72,134 articles combining 4 datasets) to expose the model to a wider variety of real news writing styles.

```
Current: ISOT only (Reuters real news)
Planned: ISOT + WELFake (Reuters + Wikipedia + diverse sources)
```

### Medium Term — Confidence Thresholding
Add an "Uncertain" verdict for borderline predictions instead of forcing binary Real/Fake:

```
> 65% real  → ✅ Real News
35–65% real → ⚠️ Uncertain — verify independently  
< 35% real  → 🚨 Fake News
```

### Long Term — BERT / Transformer Models
Fine-tune a pre-trained transformer (RoBERTa, DistilBERT) on fake news detection. Transformers understand context, not just word frequency or sequential patterns — making them far more robust to writing style variations.

```
TF-IDF + SVM → LSTM + Embeddings → BERT fine-tuned → Production
     ✅              ✅                  planned
```

---

## Conclusion

All three models achieve exceptional benchmark accuracy on the ISOT dataset. However, real-world testing reveals that high accuracy on a clean benchmark can mask poor generalization to out-of-distribution data.

The core problem is not model architecture — it is **training data diversity**. A model can only generalize to writing styles it has seen during training. Reuters-only real news creates a brittle model that mistakes any non-Reuters journalism for fake news.

The path to a truly robust fake news detector requires:
1. Diverse training data from multiple real news sources
2. Confidence thresholding to handle uncertain cases
3. Contextual models (BERT) that understand meaning, not just style

This report serves as a case study in why **evaluation beyond the benchmark is essential** in applied machine learning.

---

## Author

**Avik Sarkar**
B.Tech CSE (AI/ML) — Brainware University Barasat (2024–2028)
GitHub: [aviksarkar0204-stack](https://github.com/aviksarkar0204-stack)
Hugging Face: [Avik128](https://huggingface.co/Avik128)
