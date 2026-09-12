# 💬 FeedbackIQ — AI-Powered Customer Feedback Intelligence

> **Turn customer reviews into actionable intelligence.**

[![Live Demo](https://img.shields.io/badge/Live%20Demo-feedbackiq.streamlit.app-FF4B4B?style=for-the-badge&logo=streamlit)](https://feedbackiq.streamlit.app)
[![Model](https://img.shields.io/badge/Model-HuggingFace-FFD21E?style=for-the-badge&logo=huggingface)](https://huggingface.co/TheRock45/feedbackiq-sentiment)
[![GitHub](https://img.shields.io/badge/GitHub-rock917-181717?style=for-the-badge&logo=github)](https://github.com/rock917/AI-Powered-Review-System)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-EE4C2C?style=for-the-badge&logo=pytorch)](https://pytorch.org)

---

## 📌 Overview

FeedbackIQ is an end-to-end NLP system that analyzes Amazon product reviews using a fine-tuned **DistilBERT** transformer model to classify customer sentiment as Positive or Negative.

Built as a placement portfolio project to demonstrate complete ML engineering — from raw data to a live deployed product.

**Problem Statement:**
Businesses receiving thousands of customer reviews cannot manually read each one. FeedbackIQ automates sentiment detection, enabling product managers and analysts to understand customer satisfaction at scale — instantly.

---

## 🚀 Live Demo

**[feedbackiq.streamlit.app](https://feedbackiq.streamlit.app)**

| Feature | Description |
|---|---|
| ✨ Review Analyzer | Paste any review — get sentiment + confidence instantly |
| 📂 Batch Intelligence | Upload a CSV — analyze up to 500 reviews on cloud |
| 💡 Customer Insights | Interactive analytics dashboard with filters |
| 🤖 Model Lab | Full ML metrics, training history, model comparison |
| 🏠 Overview | Command center with KPI cards and sentiment trends |
| ℹ️ About | Project architecture and technology stack |

---

## 🏗️ System Architecture

```
Raw Amazon Reviews (568,454)
         ↓
Data Cleaning + Labeling
(4-5★ → Positive, 1-2★ → Negative, 3★ → Dropped)
         ↓
Balanced Sampling (55K + 55K = 110K reviews)
         ↓
Train / Val / Test Split (80% / 10% / 10%)
         ↓
┌──────────────────────┐    ┌──────────────────────────┐
│ TF-IDF + Logistic    │    │ Fine-tuned DistilBERT    │
│ Regression (Baseline)│    │ (distilbert-base-uncased) │
│ Accuracy: 92.12%     │    │ Accuracy: 95.01%         │
└──────────────────────┘    └──────────────────────────┘
         ↓                            ↓
         └───────────┬────────────────┘
                     ↓
            Inference Pipeline
            (src/inference.py)
                     ↓
            SQLite Inference Cache
            (SHA256 hashing)
                     ↓
            FeedbackIQ Streamlit App
                     ↓
┌───────────────────────────────────────┐
│          Deployment Stack             │
│  GitHub → Streamlit Community Cloud   │
│  Model  → Hugging Face Model Hub      │
└───────────────────────────────────────┘
```

---

## 📊 Model Performance

All metrics measured on **11,000 unseen test reviews.**

| Metric | TF-IDF + Logistic Regression | Fine-tuned DistilBERT |
|---|---|---|
| **Accuracy** | 92.12% | **95.01%** |
| **Precision** | 92.57% | **94.87%** |
| **Recall** | 91.58% | **95.16%** |
| **F1-Score** | 92.07% | **95.02%** |

**Improvement: +2.89% accuracy and +2.95% F1-Score over the TF-IDF baseline.**

### Training History

| Epoch | Train Loss | Val Loss | Train Acc | Val Acc |
|---|---|---|---|---|
| 1 | 0.2134 | 0.1344 | 91.42% | 95.03% |
| 2 | 0.1073 | 0.1426 | 96.31% | **95.55%** ← Best |
| 3 | 0.0662 | 0.1681 | 98.03% | 95.53% |

Epoch 2 selected as the best checkpoint. Val loss increased in epoch 3, indicating early overfitting. The model was automatically saved at epoch 2 using best-checkpoint tracking.

---

## 📦 Dataset

**Amazon Fine Food Reviews** — McAuley et al., Stanford University

| Property | Value |
|---|---|
| Source | [Kaggle — Amazon Fine Food Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews) |
| Total reviews | 568,454 |
| Date range | October 1999 – October 2012 |
| Unique products | 74,258 |
| Unique reviewers | 256,059 |
| Columns used | Text, Score, ProductId, Time |

### Sentiment Labeling Strategy

```
4–5 stars  →  Positive  (label = 1)
1–2 stars  →  Negative  (label = 0)
3 stars    →  Dropped   (genuinely ambiguous neutral)
```

**Why drop 3-star reviews?**
A 3-star review is genuinely ambiguous — neither clearly positive nor negative. Forcing binary labels on neutral reviews adds noise to training data and reduces model signal quality.

### Class Balancing

The raw dataset is heavily imbalanced (84% positive, 16% negative). Stratified random sampling was used to create a balanced subset:

```
Available negatives  :  82,037
Available positives  : 443,777
Sampled per class    :  55,000

Total training data  : 110,000 reviews (50% / 50%)
Train                :  88,001  (80%)
Validation           :  10,999  (10%)
Test                 :  11,000  (10%)
```

### Text Preprocessing

For DistilBERT, **light cleaning only** was applied:
- Removed HTML tags (`<br />`, `<b>`, etc.)
- Removed URLs
- Collapsed multiple whitespace
- Preserved punctuation, capitalization, and contractions

**Why not remove stopwords?**
DistilBERT understands "not good" differently from "good". Removing "not" destroys semantic meaning. Transformer models should receive minimally cleaned text.

---

## 🤖 Model Details

### Why DistilBERT?

| Property | BERT-base | DistilBERT |
|---|---|---|
| Parameters | 110M | **66M** |
| Layers | 12 | **6** |
| Speed | baseline | **60% faster** |
| Size | ~440 MB | **~250 MB** |
| Performance retained | 100% | **97%** |

DistilBERT provides near-BERT performance at a fraction of the computational cost — the correct choice for a laptop with 4GB VRAM.

### Why Not Other Architectures?

| Model | Limitation |
|---|---|
| LSTM | Sequential processing, no bidirectional context, weaker on long text |
| TF-IDF + LR | Cannot understand word order or negation ("not good" = "good not") |
| BERT-large | Too large for 4GB VRAM, significantly slower training |
| GPT-2 | Decoder-only, not designed for classification tasks |
| **DistilBERT** | Best balance of performance, speed, and memory ✅ |

### Fine-tuning Configuration

```
Model              : distilbert-base-uncased
Task               : Binary Sentiment Classification
Classes            : Negative (0), Positive (1)
Max sequence length: 256 tokens
Batch size         : 32
Learning rate      : 2e-5
Warmup             : 10% of total steps (linear warmup)
Scheduler          : Linear decay after warmup
Optimizer          : AdamW (weight decay = 0.01)
Gradient clipping  : max_norm = 1.0
Epochs             : 3 (best: epoch 2)
Hardware           : NVIDIA RTX 2050 (4GB VRAM, CUDA 12.7)
Training time      : ~7.25 hours
```

### Why max_length = 256?

```
Review length percentiles (in words):
  50th : 55 words
  75th : 96 words
  90th : 158 words
  95th : 215 words  ← 256 tokens covers up to here
  99th : 385 words

DistilBERT hard limit : 512 tokens
Our choice            : 256 tokens

Reasoning: Covers ~95% of all reviews while
halving compute time vs max_length=512.
```

### Why Learning Rate = 2e-5?

DistilBERT's weights already encode deep knowledge of English from pretraining on billions of words. A large learning rate would destroy this pretrained knowledge (catastrophic forgetting). 2e-5 gently nudges the weights toward sentiment classification without overwriting the pretrained representations.

### Tokenization Example

```
Input  : "The product quality is excellent"
Tokens : ['[CLS]', 'the', 'product', 'quality', 'is', 'excellent', '[SEP]']
IDs    : [101, 1996, 2622, 3737, 2003, 6581, 102]

[CLS] = classification token (its final vector feeds the classifier)
[SEP] = sequence separator
```

---

## ⚡ Inference Cache

To speed up repeated batch analysis, predictions are cached in a local SQLite database using SHA256 hashing:

```
For each review text:
    hash = SHA256(text.strip().lower())
        ↓
    Check SQLite: SELECT WHERE hash = ?
        ├── HIT  → return stored result instantly (<1ms)
        └── MISS → run DistilBERT → store in DB → return
```

**Benefit:** Second upload of the same CSV returns results in seconds instead of minutes. Cache hit rate is shown after every batch analysis.

**Note:** On Streamlit Community Cloud, the file system is ephemeral — the cache resets on cold starts. Cache persistence works fully in local deployment.

---

## 🛠️ Technology Stack

| Category | Technology | Version |
|---|---|---|
| Language | Python | 3.11.9 |
| Deep Learning | PyTorch (CUDA) | 2.5.1+cu121 |
| NLP | Hugging Face Transformers | 4.41.2 |
| Model | distilbert-base-uncased | 66M params |
| Baseline ML | Scikit-learn | 1.5.0 |
| Data | Pandas | 2.2.2 |
| Data | NumPy | 1.26.4 |
| Visualization | Plotly | 5.22.0 |
| Visualization | Matplotlib | 3.9.0 |
| Visualization | Seaborn | 0.13.2 |
| Web App | Streamlit | 1.35.0 |
| Cache | SQLite | built-in |
| Model Hosting | Hugging Face Hub | — |
| Deployment | Streamlit Community Cloud | — |
| Version Control | Git + GitHub | — |

---

## 📁 Project Structure

```
AI-Powered-Review-System/
│
├── app/                            # Streamlit web application
│   ├── app.py                      # Entry point + page routing
│   ├── _pages/                     # One file per page
│   │   ├── overview.py             # Command center dashboard
│   │   ├── review_analyzer.py      # Single review analysis
│   │   ├── batch_intelligence.py   # CSV batch analysis
│   │   ├── customer_insights.py    # Business intelligence
│   │   ├── model_lab.py            # ML metrics + comparison
│   │   └── about.py                # About page
│   ├── components/                 # Reusable UI components
│   │   ├── sidebar.py              # Navigation + model status
│   │   ├── charts.py               # Plotly chart functions
│   │   └── cards.py                # KPI cards, result cards
│   ├── styles/
│   │   └── theme.py                # CSS design system
│   └── utils/
│       ├── data_loader.py          # Cached data loading
│       ├── demo_data.py            # Hardcoded demo dataset
│       ├── validators.py           # Input + file validation
│       ├── cache.py                # SQLite inference cache
│       └── cached_inference.py     # Cache-aware batch inference
│
├── notebooks/                      # Jupyter experimentation
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_baseline_model.ipynb
│   ├── 04_distilbert_training.ipynb
│   └── 05_model_evaluation.ipynb
│
├── src/                            # Core ML modules
│   ├── inference.py                # Prediction pipeline
│   └── test_inference.py           # Inference verification
│
├── models/                         # Saved models (local only)
│   ├── baseline/                   # TF-IDF vectorizer + LR model
│   └── distilbert/                 # Fine-tuned DistilBERT weights
│
├── data/                           # Dataset (local only)
│   ├── raw/                        # Original Reviews.csv
│   └── processed/                  # train.csv, val.csv, test.csv
│
├── assets/                         # Generated charts + screenshots
├── requirements.txt                # Python dependencies
├── .gitignore                      # Excludes data, models, venv
└── README.md                       # This file
```

---

## ⚙️ Local Installation

### Prerequisites

- Python 3.11
- NVIDIA GPU recommended (CUDA 12.x) for training
- 8GB+ RAM
- 10GB+ free disk space



## 🌐 Deployment Architecture

```
Developer Machine (local)
        ↓  git push
GitHub Repository
        ↓  auto-deploys on every push
Streamlit Community Cloud
        ↓  downloads model at startup
Hugging Face Model Hub
(TheRock45/feedbackiq-sentiment — 267MB)
```

The trained model is hosted on Hugging Face Model Hub and loaded at runtime. This keeps the GitHub repository lightweight and follows industry best practices for ML model versioning.

### Cloud vs Local Comparison

| Feature | Cloud (Streamlit) | Local (GPU) |
|---|---|---|
| Inference | CPU | NVIDIA RTX 2050 |
| Single review | ~5-8 seconds | <1 second |
| Batch limit | 500 reviews (slider) | 10,000 reviews |
| Cache persistence | Resets on cold start | Persistent SQLite |
| Cost | Free | Local machine |

---

## ⚠️ Known Limitations

| Limitation | Description | Potential Fix |
|---|---|---|
| Labeling noise | Star rating occasionally contradicts review text | Better labeling using text + rating together |
| Truncation | Reviews over 256 tokens are cut off | Increase max_length to 512 or use sliding window |
| Domain specificity | Trained on food reviews only | Fine-tune on multi-domain dataset |
| Sarcasm | Positive words used sarcastically predicted as Positive | Add sarcasm detection pre-processing |
| CPU speed | Cloud deployment is slow for batch inference | Upgrade to paid GPU-enabled hosting |
| Cache ephemeral | SQLite resets on Streamlit Cloud cold start | Use Redis or PostgreSQL for persistence |

---

## 🔮 Future Improvements

- [ ] Multi-class sentiment — Positive / Neutral / Negative
- [ ] Aspect-based sentiment — packaging, taste, delivery separately
- [ ] SHAP-based model explainability
- [ ] Support for non-English reviews (multilingual DistilBERT)
- [ ] Persistent inference cache using Redis or PostgreSQL
- [ ] GPU-enabled deployment on paid infrastructure
- [ ] REST API endpoint for programmatic access
- [ ] Real-time streaming analysis for live review feeds
- [ ] Fine-tune on electronics, fashion, and other categories

---

## 📈 Resume Bullet Points

```
• Fine-tuned DistilBERT (66M parameters) on 110,000 balanced Amazon
  reviews achieving 95.01% accuracy and 95.02% F1-score on 11,000
  unseen test reviews — a +2.89% improvement over TF-IDF baseline

• Built complete end-to-end NLP pipeline: data cleaning, class
  balancing, tokenization, GPU fine-tuning on NVIDIA RTX 2050
  (CUDA 12.7), and evaluation using accuracy, precision, recall, F1

• Deployed production-ready ML web application at
  feedbackiq.streamlit.app using Streamlit Cloud with model hosted
  on Hugging Face Model Hub following industry MLOps practices

• Implemented SQLite inference cache using SHA256 hashing, reducing
  repeat batch analysis from minutes to seconds with real-time
  cache hit rate reporting
```

---

## 🙏 Acknowledgements

- **Dataset:** J. McAuley and J. Leskovec — *From Amateurs to Connoisseurs: Modeling the Evolution of User Expertise through Online Reviews*, Stanford University
- **Model:** Victor Sanh et al. — *DistilBERT, a distilled version of BERT*, Hugging Face
- **Framework:** Hugging Face Transformers library
- **Deployment:** Streamlit Community Cloud

---

## 👤 Author

Built by a  student as a placement portfolio project.

| | |
|---|---|
| 🌐 Live App | [feedbackiq.streamlit.app](https://feedbackiq.streamlit.app) |
| 💻 GitHub | [github.com/rock917](https://github.com/rock917) |
| 🤗 Model | [TheRock45/feedbackiq-sentiment](https://huggingface.co/TheRock45/feedbackiq-sentiment) |

---

## 📄 License

This project is for educational and portfolio purposes only.

- **Dataset:** Amazon Fine Food Reviews — J. McAuley and J. Leskovec, Stanford University
- **Base Model:** distilbert-base-uncased — Hugging Face / Victor Sanh et al.

---


