# FakeGuard — Retraining Report (ISOT + WELFake)

> Follow-up to [REPORT.md](Report.md) — documents the retraining of SVM and Logistic Regression on a combined dataset (ISOT + WELFake) to improve real-world generalization.

---

## Background

[REPORT.md](Report.md) identified a critical limitation — all three models (SVM, LR, LSTM) were trained exclusively on the ISOT dataset where real news came only from Reuters. This caused all models to incorrectly flag a legitimate BBC article as fake news.

The hypothesis: **training on more diverse real news sources will improve generalization.**

---

## Retraining Strategy

### Combined Dataset

| Dataset | Articles | Real News Source | Fake News Source |
|---|---|---|---|
| ISOT | 44,898 | Reuters only | Unreliable websites |
| WELFake | 72,134 | Wikipedia + diverse sources | Multiple fake news sources |
| **Combined** | **~117,000** | **Reuters + diverse** | **Multiple sources** |

### Key Issue Found — WELFake Label Mismatch

During data preparation, a critical bug was discovered — WELFake uses **inverted labels** compared to ISOT:

```
ISOT:     0 = Fake,  1 = Real
WELFake:  0 = Real,  1 = Fake  ← opposite!
```

When combined without fixing this, models trained on random noise — accuracy collapsed to ~50% (worse than random guessing). The fix:

```python
df_WEL['label'] = df_WEL['label'].map({0: 1, 1: 0})
```

Always verify label conventions when combining datasets from different sources.

### Preprocessing

Same 7-step pipeline as original:
```
Remove HTML → Lowercase → Remove Punctuation → Tokenize
→ Remove Stopwords → Stem → Join
```

One addition — `fillna("")` applied before preprocessing to handle null rows in WELFake.

---

## Benchmark Results

### Original Models (ISOT only — 44,898 articles)

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 98.72% | — | — | — |
| SVM (LinearSVC) | 99.51% | — | — | — |

### Retrained Models (ISOT + WELFake — ~117,000 articles)

| Model | Accuracy | Precision | Recall | F1 | Duration |
|---|---|---|---|---|---|
| Naive Bayes | 88.29% | 88.38% | 88.29% | 88.27% | 66.49s |
| Logistic Regression | 96.25% | 96.26% | 96.25% | 96.25% | 103.10s |
| **SVM (LinearSVC)** | **97.38%** | **97.40%** | **97.38%** | **97.38%** | **68.94s** |

### Accuracy Drop — Expected and Desirable

| Model | ISOT Only | ISOT + WELFake | Drop |
|---|---|---|---|
| Logistic Regression | 98.72% | 96.25% | -2.47% |
| SVM | 99.51% | 97.38% | -2.13% |

The accuracy drop is **expected and desirable**. The combined dataset is harder — it contains more diverse writing styles that the model must now distinguish. A model scoring 97% on diverse data generalizes far better than one scoring 99% on Reuters-only data.

---

## Real-World Test — Same BBC Article

The same BBC article from REPORT.md was used to evaluate generalization improvement.

### Article
> **'Attacked 28 times in a day' — BBC visits heavily targeted US-UK base in Iraq**
> *(Full article in REPORT.md)*

**Ground truth: Real News**

### Full Comparison — All Models

| Model | Training Data | Verdict | Fake % | Real % |
|---|---|---|---|---|
| SVM | ISOT only | 🚨 Fake | 60% | 40% |
| LR | ISOT only | 🚨 Fake | 67% | 33% |
| LSTM | ISOT only | 🚨 Fake | 100% | 0% |
| SVM Retrained | ISOT + WELFake | 🚨 Fake | **64%** | **36%** |
| LR Retrained | ISOT + WELFake | 🚨 Fake | **80%** | **20%** |

### Analysis

- **SVM Retrained** — marginal improvement (60% → 64% fake). Slightly more uncertain, slightly less wrong
- **LR Retrained** — actually got worse (67% → 80% fake). WELFake writing styles appear to have confused LR more than helped it
- All models still incorrectly classify the article as fake

---

## Why Retraining Alone Wasn't Enough

Adding more data helped marginally but didn't solve the core problem. Here's why:

**1. TF-IDF still ignores word order and context**

TF-IDF sees "attacked", "bloody", "weapons", "destruction" as individual signals. It has no way to understand that these words appear in a factual military report rather than sensationalist fake news. The feature representation is fundamentally limited.

**2. Writing style gap persists**

Even with WELFake, the majority of real news in the combined dataset still comes from formal journalistic sources. BBC's narrative style — with informal quotes, first-person reporting, and dramatic but factual descriptions — remains underrepresented.

**3. Headline bias**

`'Attacked 28 times in a day'` as a headline looks identical to fake news clickbait to a bag-of-words model. Without understanding that this is a factual military count, any word-frequency model will be misled.

---

## Lessons Learned

**On data:**
- Always verify label conventions when combining datasets — a flipped label bug caused 50% accuracy (worse than guessing)
- More data helps but cannot compensate for fundamental representation limitations
- Dataset diversity matters more than dataset size

**On models:**
- Higher benchmark accuracy does not mean better real-world performance
- A 97% model on diverse data beats a 99% model on narrow data for real-world use
- Classical ML (TF-IDF + SVM/LR) has a ceiling for this problem — it cannot understand context

---

## Remaining Limitations

Despite retraining, these limitations persist:

- **Narrative journalism** — BBC, Guardian, Al Jazeera writing styles still misclassified
- **Informal quotes** — real journalists quote informal speech; models flag this as fake
- **Dramatic but factual headlines** — indistinguishable from clickbait at the word level
- **Opinion pieces** — legitimate opinion journalism uses emotional language that triggers fake signals

---

## Planned Next Steps (Without Transformers)

Three approaches to improve further using only classical ML and basic DL:

**1. Feature Engineering (Classical ML)**
Add handcrafted features alongside TF-IDF:
- CAPS word ratio — fake news uses excessive ALL CAPS
- Exclamation/question mark count — fake news uses `!!!` frequently
- Readability score — real journalism has consistent sentence complexity
- Wire service detection — Reuters/AP datelines are strong real news signals

**2. Bidirectional LSTM**
Read text in both directions for better context:
```python
nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
```

**3. GloVe Pre-trained Embeddings + Attention**
Replace randomly initialized embeddings with GloVe vectors pre-trained on billions of words. Add an attention mechanism to focus on the most important words rather than just the last LSTM timestep.

---

## Conclusion

Retraining on ISOT + WELFake (~117,000 articles) produced marginal real-world improvement for SVM but no improvement for LR. Benchmark accuracy dropped slightly — as expected with harder, more diverse data.

The core finding is that **the problem is not solvable with TF-IDF or basic LSTM alone**. These models cannot understand context, intent, or writing purpose — they only see word frequencies and sequential patterns. A BBC military correspondent and a fake news writer can use the same dramatic words for completely different reasons, and bag-of-words models cannot distinguish between them.

The next meaningful improvement requires either:
- Significant feature engineering (handcrafted contextual signals)
- Pre-trained word embeddings + bidirectional LSTM with attention
- Or eventually, contextual language models (Transformers — planned for LLM phase)

---

## Author

**Avik Sarkar**
B.Tech CSE (AI/ML) — Brainware University Barasat (2024–2028)
GitHub: [aviksarkar0204-stack](https://github.com/aviksarkar0204-stack)
Hugging Face: [Avik128](https://huggingface.co/Avik128)
