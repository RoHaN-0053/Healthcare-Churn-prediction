# 🏥 Hospital Patient Churn Prediction

## 📌 Overview

This project uses machine learning to predict whether a hospital patient is likely to **churn (leave)** or remain **retained**.

The primary objective is to identify patients who are at higher risk of leaving so that they can potentially be targeted for early intervention and retention strategies.

---

## 🎯 Problem Statement

To develop a machine learning model that predicts whether a hospital patient is likely to **churn (leave)** or **remain retained**, enabling early identification of patients at risk of leaving.

---

## 📊 Dataset

The dataset contains **2,000 patient records**.

### Target Variable

| Value | Meaning  |
| ----- | -------- |
| `0`   | Retained |
| `1`   | Churned  |

The target variable is moderately imbalanced:

* **Churned:** 1,367 (68.3%)
* **Retained:** 633 (31.7%)

Because of this imbalance, accuracy alone was not used to determine model performance.

---

## 🔍 Exploratory Data Analysis

The analysis included:

* Dataset structure and data types
* Missing-value analysis
* Numerical and categorical feature analysis
* Target-variable distribution
* Feature distributions and relationships
* Identification of class imbalance

---

## ⚙️ Methodology

The following workflow was followed:

```text
Data Collection
      ↓
Data Cleaning
      ↓
Exploratory Data Analysis
      ↓
Train-Test Split
      ↓
Feature Standardization
      ↓
Model Training
      ↓
Imbalance Handling Experiments
      ↓
Model Comparison
      ↓
Final Model Selection
```

### Preprocessing

* Categorical variables were encoded where required.
* Numerical features were standardized using `StandardScaler`.
* The scaler was fitted only on the training data to prevent data leakage.
* An 80/20 train-test split was used.

---

## ⚖️ Handling Class Imbalance

Since the target variable was moderately imbalanced, multiple approaches were evaluated.

### SMOTE

SMOTE (Synthetic Minority Over-sampling Technique) was applied to the **training data only** to generate synthetic samples for the minority class.

### Class Weighting

Logistic Regression and Random Forest were also tested using:

```python
class_weight="balanced"
```

This approach increases the importance of the underrepresented class during model training without generating synthetic samples.

---

## 🤖 Models Evaluated

The following approaches were tested:

1. Logistic Regression + Standardization
2. Logistic Regression + SMOTE
3. Logistic Regression + `class_weight="balanced"`
4. Random Forest + `class_weight="balanced"`

---

## 📈 Model Comparison

| Model                                     |   Accuracy | Churn Recall | Churn F1 | Macro F1 |
| ----------------------------------------- | ---------: | -----------: | -------: | -------: |
| **Logistic Regression + Standardization** | **74.75%** |      **95%** |  **85%** |      57% |
| Logistic Regression + SMOTE               |     63.00% |          64% |      72% |  **59%** |
| Logistic Regression + Balanced Weights    |     62.50% |          62% |      71% |    **59% |
| Random Forest                             |     72.00% |          92% |      83% |      54% |
| Random Forest + Balanced Weights          |     74.00% |          96% |      84% |      52% |

> **Note:** Model selection was based primarily on the ability to identify churned patients, rather than accuracy alone.

---

## 🏆 Final Model

### Logistic Regression + Standardization

The final model selected for this project is **Logistic Regression with standardized features**.

### Performance

* **Accuracy:** 74.75%
* **Churn Precision:** 76%
* **Churn Recall:** 95%
* **Churn F1-score:** 85%
* **Macro F1-score:** 57%

### Why this model?

The primary goal is to identify patients who are likely to churn.

The final Logistic Regression model achieved **95% recall for the churn class**, meaning it correctly identified approximately 95% of actual churners in the test set.

Although SMOTE and class weighting produced slightly higher macro F1 scores, they reduced churn recall considerably. Therefore, the standard Logistic Regression model was selected based on the project's primary objective.

---

## 📌 Key Insights

* The dataset has a moderately imbalanced target variable.
* Accuracy alone can be misleading for this problem.
* The majority-class baseline is approximately **68.3% accuracy**.
* Logistic Regression achieved **74.75% accuracy**, outperforming the majority-class baseline.
* The final model achieved **95% churn recall**.
* SMOTE and class weighting were tested but did not improve churn detection.
* Random Forest was also evaluated but did not outperform Logistic Regression according to the selected evaluation criteria.

---

## 🔮 Future Improvements

Future work could include:

* Stratified cross-validation
* Gradient Boosting / XGBoost
* Feature selection
* Deployment using Flask, FastAPI, or Streamlit

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Imbalanced-learn
* Google Colab

---

## 📁 Project Structure

```text
hospital-patient-churn/
│
├── Healthcare.ipynb
├── README.md
├── dataset.csv
```

---

## 🚀 How to Run

### Clone the repository

```bash
git clone <[your-repository-url](https://github.com/RoHaN-0053/Healthcare-Churn-prediction)>
cd hospital-patient-churn
```

### Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
```

### Run the notebook

Open `Healthcare.ipynb` using Jupyter Notebook or Google Colab.

---

## 👤 Author

**Rohan Sharma**

Machine Learning / Data Science Project


