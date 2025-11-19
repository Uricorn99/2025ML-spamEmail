# Proposal 001: Data Preprocessing Pipeline for SMS Spam Classification

- **ID**: PROPOSAL-001  
- **Title**: Data Preprocessing (Cleaning, Tokenization, Vectorization)  
- **Status**: Planned  
- **Owner**: Student (Uricorn99)  
- **Created**: 2025-10-XX  

---

## 1. Motivation

The raw SMS dataset (`sms_spam_no_header.csv`) contains free-text messages that must be cleaned and converted into numerical features before model training.  
A clear and reusable preprocessing pipeline ensures:

- Consistent transformations across training, evaluation, and Streamlit app.
- Reproducibility for ML experiments.
- Fair comparison of different classifiers (Logistic Regression, Naïve Bayes, SVM).

This proposal defines the preprocessing pipeline that will be implemented in Notebook `01_preprocess.ipynb`.

---

## 2. Requirements

### 2.1 Data Loading
- Load dataset from: `data/sms_spam_no_header.csv`.
- Standardize column names (e.g., `label`, `message`).
- Check:
  - Number of rows
  - Missing values
  - Label distribution

### 2.2 Text Cleaning
- Convert all messages to lowercase.
- Remove leading/trailing whitespace.
- Normalize repeated spaces.
- (Optional) Remove punctuation.
- (Optional) Remove stopwords.
- (Optional) Apply stemming/lemmatization.

### 2.3 Train/Test Split
- Use scikit-learn `train_test_split`.
- Recommended split: 80% training / 20% testing.
- Set `random_state=42` for reproducibility.

### 2.4 Vectorization
- Use `TfidfVectorizer` from scikit-learn.
- Fit only on the training messages.
- Transform both training and testing messages.
- Consider tuning:
  - `ngram_range`
  - `min_df`, `max_df`
  - `max_features`

### 2.5 Reusability
- Save the fitted vectorizer to:
    ```
    models/tfidf_vectorizer.pkl
    ```


using `joblib.dump(...)`.

- Preprocessing logic should be reusable later in:
- Model training notebook
- Streamlit UI

### 2.6 Validation Checks
- Print:
- Dataset shape
- Label counts
- Train/test feature matrix shapes
- Handle missing/empty messages by filling with empty string or skipping.

---

## 3. Out of Scope

These will NOT be included in this proposal:

- Advanced text normalization (lemmatization trees, complicated regex)
- Deep learning models
- Class imbalance handling (SMOTE, oversampling)
- Hyperparameter tuning
- Deployment configuration

These may be added in future proposals if needed.

---

## 4. Acceptance Criteria

This proposal is considered **Done** when:

1. A new notebook `notebooks/01_preprocess.ipynb` contains code for:
 - Data loading  
 - Cleaning  
 - Train/test split  
 - TF-IDF vectorization  

2. `models/tfidf_vectorizer.pkl` is successfully generated and can be reloaded without errors.

3. Running the preprocessing notebook produces:
 - Summary statistics  
 - TF-IDF feature matrices  
 - No missing-value errors  
 - Consistent shapes for train/test splits  

4. The vectorizer can be imported by later notebooks and the Streamlit UI.

---

## 5. Implementation Plan

1. Create notebook `01_preprocess.ipynb`.
2. Load CSV and inspect dataset.
3. Implement text cleaning functions.
4. Apply cleaning to all messages.
5. Split dataset using `train_test_split`.
6. Fit `TfidfVectorizer` on training data.
7. Transform train/test data.
8. Save vectorizer to `models/`.
9. Update proposal status to **In Progress** and then **Done** once confirmed.
