# Project: SMS Spam Email Classification (Homework 3)

## 1. Overview

This project implements an end-to-end SMS spam classification system as Homework 3 for the Machine Learning course.

The goal is to reproduce and extend the pipeline from Chapter 3 of *Hands-On Artificial Intelligence for Cybersecurity*, using the OpenSpec (Spec-Driven Development) workflow.

The system loads the `sms_spam_no_header.csv` dataset, performs data preprocessing, trains several machine learning models, evaluates them with multiple metrics and visualizations, and exposes a simple Streamlit web UI for interactive predictions.

## 2. Objectives

- Build a reproducible ML pipeline for SMS spam detection.
- Practice OpenSpec-driven development (project context + proposals + AGENTS workflow).
- Experiment with different classifiers (Logistic Regression, Naïve Bayes, SVM).
- Provide clear evaluation (metrics + charts).
- Deploy a minimal but functional Streamlit demo app.

## 3. Dataset

- **Source**: Packt book repository – Chapter03/datasets/sms_spam_no_header.csv  
- **Task**: Binary classification  
  - `label`: spam / ham (or equivalent)  
  - `message`: SMS text

Assumptions:

- Dataset is small enough to fit in memory.
- No severe class imbalance issues that require advanced rebalancing beyond simple checks.

## 4. Tech Stack

- **Language**: Python 3.x
- **Core Libraries**:
  - `pandas` – data loading and preprocessing
  - `scikit-learn` – vectorization (e.g. `TfidfVectorizer`) and classifiers
  - `numpy` – numerical operations
  - `matplotlib` / `seaborn` – charts, confusion matrix, ROC/PR curves
  - `streamlit` – web UI for demo
  - `joblib` – saving/loading trained models and vectorizers

## 5. Project Structure (planned)

```text
2025ML-spamEmail/
├── openspec/
│   ├── project.md
│   ├── AGENTS.md
│   └── proposals/
│       └── 001-data-preprocessing.md
├── data/
│   └── sms_spam_no_header.csv
├── notebooks/
│   ├── 01_preprocess.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_evaluation.ipynb
├── models/
│   ├── tfidf_vectorizer.pkl
│   └── best_model.pkl
├── app.py               # Streamlit app
├── requirements.txt
├── README.md
└── .gitignore
```
## 6. Core Pipeline (planned)

### 6.1 Data Loading
- Load CSV from `data/sms_spam_no_header.csv`.
- Perform basic sanity checks:
  - Check number of rows.
  - Check missing values.
  - Check label distribution.

### 6.2 Preprocessing
- Convert text to lowercase.
- Remove unnecessary whitespace and basic punctuation.
- (Optional) Remove stopwords.
- (Optional) Apply simple stemming or lemmatization.
- Split into train / validation / test (or train / test) using `train_test_split`.

### 6.3 Vectorization
- Use `TfidfVectorizer` from scikit-learn.
- Fit vectorizer on training data only.
- Transform both train and test messages.
- Consider tuning:
  - `ngram_range`
  - `min_df`
  - `max_df`
  - `max_features`

### 6.4 Model Training
Train at least the following models:

1. **Logistic Regression**
2. **Multinomial Naïve Bayes**
3. **Linear SVM** (e.g. `LinearSVC`)

- Compare performance across models.
- Select the best performing model (according to F1 or accuracy).

### 6.5 Evaluation & Visualization
- Compute evaluation metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-score
- Generate visualizations:
  - Confusion matrix (heatmap)
  - ROC curve (for probability-based models)
  - Precision–Recall curve
- Perform simple interpretability analysis:
  - Most frequent spam words
  - Important TF-IDF features

### 6.6 Streamlit UI
- Load the saved TF-IDF vectorizer and trained model from `models/`.
- Provide a text input box for user to enter an SMS message.
- Display:
  - Predicted label (Spam / Ham)
  - Prediction probability or decision score
- Optional enhancements:
  - Highlight important keywords
  - Show model metadata

---

## 7. Conventions

### 7.1 Code Style
- Follow PEP 8 where reasonable.
- Use clear function names and comments.

### 7.2 Notebooks
- Store notebooks in `notebooks/`.
- Use numeric prefixes to show order:
  - `01_preprocess.ipynb`
  - `02_model_training.ipynb`
  - `03_evaluation.ipynb`

### 7.3 File Paths
- Use relative paths from project root:
  - `data/...`
  - `models/...`
  - `notebooks/...`

### 7.4 Random Seed
- Use a fixed seed (`random_state=42`) for reproducibility.

---

## 8. Risks / Non-Goals

- No extensive hyperparameter tuning.
- No advanced deployment tools (e.g., Docker, CI/CD pipelines).
- No deep learning models.
- Explainability will remain simple (feature importance, TF-IDF inspection).

---

## 9. OpenSpec Usage

- Use `openspec/proposals/*.md` for every significant project feature:
  - Data preprocessing
  - Model training & comparison
  - Evaluation and visualizations
  - Streamlit UI integration
- Each proposal must:
  - Describe motivation
  - Define requirements
  - Set acceptance criteria
  - Track implementation status

