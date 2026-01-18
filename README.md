# Machine Learning Assignment 2  
## Classification Models – From Scratch Implementation

---

## 1. Problem Statement
The objective of this assignment is to design, implement, evaluate, and deploy multiple
**classification algorithms from scratch** on a real-world dataset.

The project demonstrates a complete **end-to-end machine learning workflow**, including:
- Data preprocessing
- Algorithm implementation from scratch
- Model evaluation using standard metrics
- Model comparison
- Interactive deployment using Streamlit

---

## 2. Dataset Description
> 📌 To be updated after final dataset confirmation

- **Dataset Name**: TODO  
- **Source**: TODO (Kaggle / UCI link)  
- **Problem Type**: Binary / Multi-class Classification  
- **Number of Instances**: TODO (≥ 500)  
- **Number of Features**: TODO (≥ 12)  
- **Target Variable**: `loan_status`  

### Feature Overview
- Numerical Features: TODO  
- Categorical Features: TODO  

### Preprocessing Overview
- Handling missing values
- Encoding categorical variables
- Feature scaling
- Train-test split

---

## 3. Models Implemented (From Scratch)
All models listed below are implemented **without using predefined machine learning models**.

1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbors (KNN)  
4. Naive Bayes Classifier  
5. Random Forest (Ensemble)  
6. XGBoost (Ensemble)  

> 📌 Each model will be added incrementally.

---

## 4. Evaluation Metrics
Each model is evaluated using the following metrics:

- Accuracy  
- AUC Score  
- Precision  
- Recall  
- F1 Score  
- Matthews Correlation Coefficient (MCC)  

---

## 5. Model Comparison Table

| ML Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|--------|----------|-----|-----------|--------|----------|-----|
| Logistic Regression | TODO | TODO | TODO | TODO | TODO | TODO |
| Decision Tree | TODO | TODO | TODO | TODO | TODO | TODO |
| KNN | TODO | TODO | TODO | TODO | TODO | TODO |
| Naive Bayes | TODO | TODO | TODO | TODO | TODO | TODO |
| Random Forest | TODO | TODO | TODO | TODO | TODO | TODO |
| XGBoost | TODO | TODO | TODO | TODO | TODO | TODO |

---

## 6. Model Performance Observations

| ML Model | Observation |
|--------|-------------|
| Logistic Regression | TODO |
| Decision Tree | TODO |
| KNN | TODO |
| Naive Bayes | TODO |
| Random Forest | TODO |
| XGBoost | TODO |

---

## 7. Streamlit Web Application
An interactive Streamlit application is developed with the following features:

- CSV dataset upload (test data only)
- Model selection dropdown
- Display of evaluation metrics
- Confusion matrix / classification report visualization

---

## 8. Project Structure
ML_assignment2/
│-- app.py
│-- requirements.txt
│-- README.md
│-- model/
│ ├── logistic.py
│ ├── decision_tree.py
│ ├── knn.py
│ ├── naive_bayes.py
│ ├── random_forest.py
│ ├── xgboost.py

---

## 9. Deployment
The Streamlit application will be deployed using **Streamlit Community Cloud**.

- **GitHub Repository**: TODO  
- **Live Streamlit App**: TODO  

---

## 10. How to Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py