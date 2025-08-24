# Heart Disease Prediction – Machine Learning Pipeline

This repository implements a **full ML pipeline** on the [UCI Heart Disease dataset](https://archive.ics.uci.edu/ml/datasets/heart+Disease), covering **data preprocessing, PCA, feature selection, supervised & unsupervised modeling, hyperparameter tuning, and deployment options (Streamlit + Ngrok)**.

---

## 📦 What’s Included
- Data preprocessing scripts/notebooks
- Trained models in `.pkl` format
- Step-by-step Jupyter notebooks for each stage
- Optional Streamlit UI for interactive predictions
- `requirements.txt` for one-command environment setup

---

## 📂 Repository Structure
```
Heart_Disease_Project/
├─ data/
│   └─ heart_disease.csv                 # dataset (add your copy here or load via notebook)
├─ notebooks/
│   ├─ 01_data_preprocessing.ipynb
│   ├─ 02_pca_analysis.ipynb
│   ├─ 03_feature_selection.ipynb
│   ├─ 04_supervised_learning.ipynb
│   ├─ 05_unsupervised_learning.ipynb
│   └─ 06_hyperparameter_tuning.ipynb
├─ models/
│   └─ final_model.pkl                   # exported best model
├─ ui/
│   └─ app.py                            # Streamlit UI (optional)
├─ deployment/
│   └─ ngrok_setup.txt                   # notes/commands to expose Streamlit
├─ results/
│   └─ evaluation_metrics.txt
├─ requirements.txt
├─ README.md
└─ .gitignore
```

> Tip: keep `data/raw` out of Git if it’s large/private; commit only small samples or instructions to fetch.

---

## ⚙️ Setup

### 1) Clone
```bash
git clone https://github.com/MahmoudHegazy-01/Heart_Disease_Project.git
cd <Heart_Disease_Project>
```


### 2) Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Reproducible Workflow

### A) Data Preprocessing & EDA
Run the notebook to clean data (missing values, encoding, scaling) and explore distributions.
```bash
jupyter notebook notebooks/01_data_preprocessing.ipynb
```

### B) PCA (Dimensionality Reduction)
Compute principal components and review variance explained.
```bash
jupyter notebook notebooks/02_pca_analysis.ipynb
```

### C) Feature Selection
Obtain RF importance, RFE.
```bash
jupyter notebook notebooks/03_feature_selection.ipynb
```

### D) Supervised Models
Train Logistic Regression, Decision Tree, Random Forest, SVM. Log metrics (Accuracy, Precision, Recall, F1, ROC-AUC).
```bash
jupyter notebook notebooks/04_supervised_learning.ipynb
```

### E) Unsupervised Models
Run K-Means (Elbow) and Hierarchical Clustering (Dendrogram) and compare to labels.
```bash
jupyter notebook notebooks/05_unsupervised_learning.ipynb
```

### F) Hyperparameter Tuning
Use `GridSearchCV` to optimize and export the best pipeline to `/models/final_model.pkl`.
```bash
jupyter notebook notebooks/06_hyperparameter_tuning.ipynb
```

---


---


## 🧾 Dataset
- UCI Heart Disease: https://archive.ics.uci.edu/ml/datasets/heart+Disease

---

## 🧰 Troubleshooting
- If you see version conflicts, try upgrading pip and reinstall:
  ```bash
  python -m pip install --upgrade pip
  pip install -r requirements.txt --upgrade
  ```
- For Apple Silicon or CUDA setups, prefer platform-specific wheels.

---

## 🤝 Contributing
Pull requests are welcome. For major changes, open an issue to discuss the plan.

