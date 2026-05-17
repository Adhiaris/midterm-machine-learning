# 🤖 Machine Learning Midterm Projects

A collection of three machine learning notebooks covering **Clustering**, **Regression**, and **Fraud Detection (Classification)**. Each notebook applies end-to-end ML pipelines including preprocessing, model training, hyperparameter tuning, experiment tracking, and explainability.

> ⚠️ **Note:** This repository covers **Machine Learning models only**. Deep Learning experiments (Neural Network models) exist in a separate repository.

---

## 📁 Repository Structure

```
├── Midterm-Clustering.ipynb           # Customer/data segmentation using KMeans
├── midterm-regresin-updated.ipynb     # Regression with Ridge (Optuna + MLflow + LIME)
└── midterm-fraud-detection.ipynb      # Fraud detection using Random Forest classifier
```

---

## 📌 Project Overview

### 1. 📊 Clustering — `Midterm-Clustering.ipynb`
Unsupervised learning notebook that segments data into meaningful groups.

- **Model:** KMeans Clustering
- **Evaluation:** Silhouette Score, Elbow Method
- **Tuning:** Optuna for optimal `k`
- **Libraries:** `scikit-learn`, `optuna`, `matplotlib`

---

### 2. 📈 Regression — `midterm-regresin-updated.ipynb`
Supervised regression task predicting a continuous target variable from 89 features.

- **Model:** Ridge Regression
- **Tuning:** Optuna (best alpha search, 50 trials)
- **Metrics:**

| Metric | Value  |
|--------|--------|
| MSE    | 70.14  |
| RMSE   | 8.38   |
| MAE    | 6.05   |
| R²     | 0.357  |

- **Experiment Tracking:** MLflow (`midterm-regression` experiment)
- **Explainability:** LIME (Local Interpretable Model-agnostic Explanations)
- **Preprocessing:** Median imputation, IQR-based outlier removal, StandardScaler
- **Split:** 80/20 train-test

---

### 3. 🔍 Fraud Detection — `midterm-fraud-detection.ipynb`
Binary classification task detecting fraudulent financial transactions.

- **Model:** Random Forest Classifier
- **Dataset:** IEEE-CIS Fraud Detection (50% sample, stratified split)
- **Imbalance Handling:** SMOTE (oversampling)
- **Tuning:** Optuna (`n_estimators`, `max_depth`, `min_samples_split`)
- **Metrics:**

| Metric         | Value  |
|----------------|--------|
| AUC-ROC        | 0.8776 |
| Accuracy       | 0.92   |
| F1 (Fraud)     | 0.34   |

- **Experiment Tracking:** MLflow (`midterm-fraud-detection` experiment)
- **Preprocessing:** LabelEncoding, Median imputation, float32/int32 type casting

---

## 🛠️ Tech Stack

| Tool               | Purpose                            |
|--------------------|------------------------------------|
| `scikit-learn`     | ML models, preprocessing, metrics |
| `imbalanced-learn` | SMOTE for class imbalance          |
| `optuna`           | Hyperparameter optimization        |
| `mlflow`           | Experiment tracking & model logging|
| `lime`             | Model explainability               |
| `pandas`, `numpy`  | Data manipulation                  |
| `matplotlib`       | Visualization                      |

---

## 📂 Datasets

| Notebook        | Dataset                                              |
|-----------------|------------------------------------------------------|
| Clustering      | Custom unlabeled dataset                             |
| Regression      | `midterm-regresi-dataset.csv` (87,557 rows × 90 cols)|
| Fraud Detection | `train_transaction.csv` (IEEE-CIS Fraud Detection)   |

---

## 📊 MLflow Experiments

| Experiment Name            | Model             | Key Metric       |
|----------------------------|-------------------|------------------|
| `midterm-regression`       | Ridge Regression  | R² = 0.357       |
| `midterm-fraud-detection`  | Random Forest     | AUC-ROC = 0.878  |

---

## 🚀 How to Navigate

1. **Clone the repository** and open the notebooks in [Google Colab](https://colab.research.google.com/) or JupyterLab.
2. Each notebook is **self-contained** — install dependencies by running the `!pip install` cell at the top.
3. Notebooks follow this general structure:
   - 📦 Library installation & imports
   - 📂 Dataset loading & preprocessing
   - 🤖 **Machine Learning Model** section (focus of this repo)
   - 📊 Evaluation metrics & visualizations
   - 🧪 MLflow experiment logging

> 💡 Cells marked **`Deep Learning Model`** are **excluded from this repository's scope** and are available in the Deep Learning repository.
