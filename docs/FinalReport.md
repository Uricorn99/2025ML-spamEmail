# Final Report — SMS Spam Email Classifier (Phase 1 → Phase 4)

This report summarizes the end‑to‑end work completed in this project: preprocessing, model development, evaluation, and a full Streamlit dashboard for interactive spam detection.

Repository: https://github.com/Uricorn99/2025ML-spamEmail  
Demo (Streamlit Cloud): https://2025ml-spamemail-uricorn.streamlit.app/

---

## Contents
- Phase 1 — Baseline Classifier
- Phase 2 — Improve Recall
- Phase 3 — Restore Precision with High Recall
- Phase 4 — Visualization + Streamlit Dashboard
- Artifacts & Directory Map
- Reproducibility & Validation
- Appendix — Quick Commands

---

## Phase 1 — Baseline Classifier

Goal: Build a simple and reproducible binary classifier using TF‑IDF + linear models.

### Deliverables
- Preprocessing: notebooks/01_preprocess.ipynb  
- Training: notebooks/02_model_training.ipynb  
- Prediction: notebooks/03_evaluation.ipynb  
- Model artifacts:
    - models/tfidf_vectorizer.pkl
    - models/best_model.pkl
    - models/train_test_text.pkl
    - models/train_test_tfidf.pkl

### Preprocessing Steps
- Lowercasing  
- Punctuation removal  
- Whitespace normalization  
- Train/test split (80/20, random_state=42)  
- TF‑IDF vectorization  

### Training
- Logistic Regression (best baseline)
- Naive Bayes
- Linear SVM

### Baseline Metrics (example)
Accuracy ~0.97  
Precision high  
Recall ~0.85 (recall weakness fixed later)

---

## Phase 2 — Improve Recall

Goal: Detect more spam (boost recall).

### Improvements
- TF‑IDF tuning: ngram_range, min_df
- Logistic Regression tuning
- Class‑weight = balanced
- Probability threshold adjustments

### Observed Example
Accuracy: 0.973  
Precision: 0.852  
Recall: 0.966  
F1: 0.905

Recall target achieved.

---

## Phase 3 — Restore Precision with High Recall

Goal: Precision ≥ 0.90 AND Recall ≥ 0.93.

### Recommended Settings
ngram_range = (1,2)  
min_df = 2  
sublinear_tf = True  
C = 2.0  
threshold = 0.50  
class_weight = balanced  

### Observed Results
Accuracy: 0.9848  
Precision: 0.9231  
Recall: 0.9664  
F1: 0.9443  

Both precision and recall targets met.

---

## Phase 4 — Visualization + Streamlit App

### CLI Visualizations (Evaluation Notebook)
- Class distribution
- Token frequency (per class)
- Confusion matrix
- ROC curve
- Precision–Recall curve
- Threshold sweep

### Streamlit Dashboard (app.py)
Features:
- Dataset & column pickers
- Class distribution summary
- Top tokens by class
- Confusion matrix
- ROC & PR curves
- Threshold sweep table
- Live inference (with spam probability bar)
- Example buttons (Spam / Ham)

Deployed at:  
https://github.com/Uricorn99/2025ML-spamEmail

---

## Artifacts & Directory Map

```
2025ML-spamEmail/
  data/
      sms_spam_no_header.csv
  models/
      tfidf_vectorizer.pkl
      best_model.pkl
      train_test_text.pkl
      train_test_tfidf.pkl
  notebooks/
      01_preprocess.ipynb
      02_model_training.ipynb
      03_evaluation.ipynb
  openspec/
      project.md
      AGENTS.md
      proposals/
          001-data-preprocessing.md
  app.py
  requirements.txt
  README.md
```

---

## Reproducibility & Validation

- Deterministic preprocessing (fixed seed)
- TF‑IDF vectorizer saved for reuse
- Best model saved as pickle
- Notebook outputs consistent across runs
- Streamlit input normalization aligned with training steps

---

## Appendix — Quick Commands

### Preprocess
```
jupyter notebook notebooks/01_preprocess.ipynb
```

### Train
```
jupyter notebook notebooks/02_model_training.ipynb
```

### Evaluate
```
jupyter notebook notebooks/03_evaluation.ipynb
```

### Launch Streamlit
```
streamlit run app.py
```

---

All four phases are complete. The project is fully reproducible, documented, and deployed.
