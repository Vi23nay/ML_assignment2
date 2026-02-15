# Machine Learning Assignment 2
## Classification Models — Implementation, Evaluation & Deployment

---

## 1. Problem Statement

The objective of this assignment is to design, implement, evaluate, and deploy multiple
**classification algorithms** on a real-world dataset.

The project demonstrates a complete **end-to-end machine learning workflow**, including:
- Data preprocessing (encoding, scaling, stratified splitting)
- Implementation of 6 classification models
- Model evaluation using 6 standard metrics
- Model comparison
- Interactive deployment using Streamlit

---

## 2. Dataset Description

- **Dataset Name**: Loan Approval Prediction
- **Source**: Kaggle (Public Repository)
- **Problem Type**: Binary Classification
- **Number of Instances**: 45,000
- **Number of Features**: 13 (excluding Target)
- **Target Variable**: `Target` (0 = Not Approved, 1 = Approved)

### Feature Overview
- **Numerical Features (8):** person_age, person_income, person_emp_exp, loan_amnt, loan_int_rate, loan_percent_income, cb_person_cred_hist_length, credit_score
- **Categorical Features (5):** person_gender, person_education, person_home_ownership, loan_intent, previous_loan_defaults_on_file

### Preprocessing Steps
- Label Encoding of categorical variables
- Stratified 80/20 train-test split (random_state=42)
- Standard Scaling (StandardScaler)
- Class imbalance handling:
  - `class_weight="balanced"` for Logistic Regression, Decision Tree, Random Forest
  - SMOTE oversampling for KNN
  - Balanced priors `[0.5, 0.5]` for Naive Bayes
  - `scale_pos_weight` (auto-computed) for XGBoost

### Class Distribution
- **Class 0** (Not Approved): 35,000 samples (77.8%)
- **Class 1** (Approved): 10,000 samples (22.2%)
- **Imbalance Ratio**: 3.5 : 1

---

## 3. Models Implemented

All models are implemented with wrapper classes in the `model/` directory. Each class provides `fit()`, `predict()`, `predict_proba()`, and `score()` methods.

| # | Model | File | Key Parameters |
|---|-------|------|----------------|
| 1 | Logistic Regression | `model/logistic.py` | solver=lbfgs, max_iter=5000, class_weight=balanced |
| 2 | Decision Tree Classifier | `model/decision_tree.py` | criterion=entropy, max_depth=15, class_weight=balanced |
| 3 | K-Nearest Neighbors (KNN) | `model/knn.py` | n_neighbors=7, weights=distance, SMOTE oversampling |
| 4 | Naive Bayes (Gaussian) | `model/naive_bayes.py` | var_smoothing=1e-7, priors=[0.5, 0.5] |
| 5 | Random Forest (Ensemble) | `model/random_forest.py` | n_estimators=200, max_depth=15, class_weight=balanced |
| 6 | XGBoost (Ensemble) | `model/xgboost_model.py` | n_estimators=200, max_depth=6, lr=0.05, scale_pos_weight=auto |

---

## 4. Evaluation Metrics

Each model is evaluated using the following metrics:

| Metric | Description |
|--------|-------------|
| Accuracy | Proportion of correct predictions over all predictions |
| AUC | Area Under the ROC Curve — discrimination ability across thresholds |
| Precision | Proportion of true positives among all positive predictions |
| Recall | Proportion of actual positives correctly identified |
| F1 Score | Harmonic mean of Precision and Recall |
| MCC | Matthews Correlation Coefficient — ranges from -1 to +1 |

---

## 5. Model Comparison Table

All models evaluated on 20% test split (threshold = 0.5):

| ML Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|----------|----------|-----|-----------|--------|----------|-----|
| Logistic Regression | 0.8487 | 0.9514 | 0.6051 | 0.9185 | 0.7295 | 0.6570 |
| Decision Tree | 0.8863 | 0.9360 | 0.6974 | 0.8630 | 0.7714 | 0.7039 |
| KNN | 0.8562 | 0.9304 | 0.6246 | 0.8845 | 0.7322 | 0.6559 |
| Naive Bayes | 0.7367 | 0.9417 | 0.4576 | 0.9985 | 0.6276 | 0.5493 |
| Random Forest (Ensemble) | 0.9133 | 0.9738 | 0.7654 | 0.8795 | 0.8185 | 0.7651 |
| XGBoost (Ensemble) | 0.9054 | 0.9773 | 0.7263 | 0.9220 | 0.8125 | 0.7600 |

---

## 6. Model Performance Observations

| ML Model | Observation about Model Performance |
|----------|-------------------------------------|
| Logistic Regression | Strong recall (0.9185) and AUC (0.9514). Linear boundary captures class separation effectively with balanced weights. Good baseline performer. |
| Decision Tree | Good accuracy (0.8863) with highest precision among non-ensemble models (0.6974). Entropy criterion with balanced weights provides solid decision boundaries. |
| KNN | SMOTE + k=7 gives recall 0.8845. Non-parametric instance-based approach performs well on 13 features with distance weighting. |
| Naive Bayes | Highest recall (0.9985) with balanced priors but lowest precision (0.4576). Conditional independence assumption captures class-conditional distributions well, AUC = 0.9417. |
| Random Forest (Ensemble) | Best accuracy (0.9133) and best F1 (0.8185). Ensemble of 200 trees with balanced weights provides the most balanced precision-recall trade-off. Best overall performer. |
| XGBoost (Ensemble) | Best AUC (0.9773) and 2nd best recall (0.9220). Gradient boosting with auto-computed scale_pos_weight effectively handles 3.5:1 imbalance. Strong overall performer. |

---

## 7. Streamlit Web Application

An interactive Streamlit web application is developed with the following features:

- **a.** CSV dataset upload option — upload test data via sidebar file uploader
- **b.** Model selection dropdown — choose from all 6 implemented models
- **c.** Display of evaluation metrics — Accuracy, AUC, Precision, Recall, F1, MCC
- **d.** Confusion matrix and classification report visualization
- Compare All Models tab — evaluates all 6 models side-by-side
- ROC curves, feature importance charts, grouped bar charts
- Adjustable classification threshold slider
- Download training data button

---

## 8. Project Structure

```
ML_assignment2/
├── app.py                    # Streamlit web application
├── loan_data.csv             # Dataset
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
└── model/
    ├── __init__.py            # Package init
    ├── logistic.py            # Logistic Regression
    ├── decision_tree.py       # Decision Tree Classifier
    ├── knn.py                 # K-Nearest Neighbors
    ├── naive_bayes.py         # Naive Bayes (Gaussian)
    ├── random_forest.py       # Random Forest (Ensemble)
    └── xgboost_model.py       # XGBoost (Ensemble)
```

---

## 9. Deployment

The Streamlit application is deployed using **Streamlit Community Cloud**.

- **GitHub Repository**: [Link](https://github.com/Vi23nay/ML_assignment2)
- **Live Streamlit App**: [Link](https://<your-app>.streamlit.app)

---

## 10. How to Run Locally

```bash
git clone <repo-url>
cd ML_assignment2
pip install -r requirements.txt
streamlit run app.py
```
