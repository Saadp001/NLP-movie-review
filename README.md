# 🎬 IMDB Sentiment Analysis — From Word Counting to BERT

> *Can a model that counts words beat one trained on billions of sentences? Spoiler: yes, if you use the powerful one wrong.*

---

## 📊 Results at a Glance

| Stage | Representation | Model | Accuracy |
|:---:|:---|:---|:---:|
| 1 | Bag of Words | Logistic Regression | 87–88% |
| 2 | TF-IDF | Logistic Regression | 89–91% |
| 3 | GloVe Embeddings | Logistic Regression | **78%** ⬇️ |
| 4 | GloVe Embeddings | LSTM | 88.4% |
| 5 | BERT (fine-tuned) | Transformer | **92.6%** 🏆 |

> **The twist:** Stage 3 uses word vectors trained on billions of words — and scores *worse* than counting words. [Here's why](#-the-key-insight).

---

## 🗂️ Project Structure

```
imdb-sentiment-analysis/
│
├── notebooks/
│   ├── 01_eda.ipynb                  # Exploratory data analysis
│   ├── 02_bow_model.ipynb            # Bag of Words + Logistic Regression
│   ├── 03_tfidf_model.ipynb          # TF-IDF + Logistic Regression
│   ├── 04_glove_lr_model.ipynb       # GloVe Embeddings + Logistic Regression
│   ├── 05_lstm_model.ipynb           # GloVe Embeddings + LSTM
│   └── 06_bert_model.ipynb           # Fine-tuned BERT (run on Colab)
│
├── data/
│   └── IMDB Dataset.csv              # 50K reviews from Kaggle
│
├── requirements.txt
└── README.md
```

---

## 🧠 The Approach

Same dataset. Same labels. **Only the text representation changes.**

This is a controlled experiment — every accuracy difference is purely because of the technique, not the data.

```
Text → numbers → model → prediction
        ↑
   This is the only thing that changes
   across all 5 stages
```

---

## 🔍 Stage by Stage

### Stage 1 — Bag of Words + Logistic Regression
Count how many times each word appears. That's it.

```python
vectorizer = CountVectorizer(max_features=10000)
X_train_bow = vectorizer.fit_transform(X_train)
X_test_bow = vectorizer.transform(X_test)      # same vocab, no leakage
```

- ✅ Simple, fast, interpretable
- ❌ "terrible" and "awful" are completely unrelated
- ❌ Common words like "the" dominate signal
- ❌ Word order lost — "not good" = "good not"

---

### Stage 2 — TF-IDF + Logistic Regression
Weight words by how rare they are across all documents.

```
TF-IDF = how often word appears here × how rare it is everywhere
```

- "the" → appears everywhere → IDF → ~0 → score collapses
- "terrible" → rare → high IDF → score amplified

**What changed in the learned weights:**
```
BoW  top positive:  bimbo, disappoint, dismiss      ← noise
TFIDF top positive: great, excellent, brilliant     ← real signal
```

---

### Stage 3 — GloVe + Logistic Regression ⚠️

Words as dense vectors in geometric space:
```
"terrible" → [−0.82,  0.14, −0.53, ...]
"awful"    → [−0.79,  0.11, −0.51, ...]   ← neighbours
"great"    → [ 0.76,  0.91,  0.44, ...]   ← opposite direction
```

**The problem:** averaging all word vectors into one destroys everything.

```python
# this one line caused 78% accuracy
return np.mean(vectors, axis=0)
```

Positive + negative vectors cancel → model is confused → 78%.

---

### Stage 4 — GloVe + LSTM
Read word vectors *sequentially* instead of averaging them.

```
Architecture:
Input → Embedding (GloVe) → LSTM (128 units) → Dense (sigmoid)
Params:   1,000,000 frozen      117,248 trained     129 trained
```

Same GloVe embeddings. Right model. 78% → 88.4%.

The LSTM reads "not" → carries it in memory → reads "good" → understands the negation. No averaging, no signal loss.

---

### Stage 5 — Fine-tuned BERT 🏆
Every word attends to every other word *simultaneously*.

```
LSTM:        word1 → word2 → word3 → ... (sequential, memory fades)
Transformer: all words look at all words at once (parallel, no forgetting)
```

BERT was pretrained on:
- Entire Wikipedia
- 800 million words of books

Fine-tuning adds one classification layer and gently nudges 109M parameters toward sentiment.

```python
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=2
)
# 109,483,778 parameters — almost all already trained
```

Training for just 3 epochs achieves 92.6%.

---

## 💡 The Key Insight

**Why did 78% (GloVe) < 91% (TF-IDF)?**

GloVe embeddings are genuinely more powerful than word counts. The geometry is real.

But averaging 200 word vectors:
- Cancels positive and negative sentiment vectors
- Destroys word order — "not good" = "good not"
- Loses which words modified which other words

> *A powerful representation used badly beats a simple one used well.*

TF-IDF keeps each word's signal separate. Dumber about meaning, smarter about not throwing information away.

---

## 📈 Parameter Comparison

```
Logistic Regression  →         10,001 params
LSTM model           →        117,377 params
BERT                 →  109,483,778 params  (932× larger than LSTM)
```

BERT's 109M parameters were already trained before touching IMDB. That's transfer learning.

---

## 🛠️ Tech Stack

```
Python 3.10+
Scikit-learn        — BoW, TF-IDF, Logistic Regression
TensorFlow / Keras  — LSTM
PyTorch             — BERT training loop
HuggingFace         — bert-base-uncased
GloVe               — Pretrained word vectors (Stanford NLP)
Google Colab        — T4 GPU for BERT
```

---

## ⚙️ Setup

```bash
git clone https://github.com/yourusername/imdb-sentiment-analysis
cd imdb-sentiment-analysis
pip install -r requirements.txt
```

Download the dataset from [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) and place it in the `data/` folder.

For Stage 5 (BERT) — open `06_bert_model.ipynb` in [Google Colab](https://colab.research.google.com) with a T4 GPU runtime.

---

## 📦 Requirements

```
pandas
numpy
scikit-learn
tensorflow
torch
transformers
matplotlib
seaborn
```

---

## 📝 Full Writeup

Read the full article on Medium: [link]

---

## 🤝 Connect

Built by **Saad** — ML engineer in progress.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)](https://www.linkedin.com/in/saad-patel-469016314/)
[![Medium](https://img.shields.io/badge/Medium-Article-black?style=flat&logo=medium)](https://medium.com/@psaad2988/i-built-5-nlp-models-on-the-same-dataset-the-most-powerful-one-scored-worst-heres-why-25346a41e961?postPublishedType=initial)

---

*If this helped you, drop a ⭐ on the repo.*
