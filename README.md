# Shopping ML Project

## Project Overview
This project explores the **shift between online and in-store shopping behavior** using a real-world-inspired Kaggle dataset.  
It demonstrates the application of multiple classical machine learning algorithms for **classification tasks**, including:

- Logistic Regression  
- Support Vector Machine (SVM)  
- K-Nearest Neighbors (KNN)  
- Decision Tree  
- Random Forest  
- Gradient Boosting (XGBoost/LightGBM)  
- Naive Bayes  

The goal is to **analyze consumer shopping preferences**, perform **EDA**, build predictive models, and interpret the results.

---

## Dataset
**Source:** [Kaggle: Online vs In-Store Shopping Behaviour Dataset](https://www.kaggle.com/datasets/shree0910/online-vs-in-store-shopping-behaviour-dataset)  

- **Size:** 11,789 records, 26 features  
- **Features:** Demographics, digital habits, shopping patterns, logistics preferences, psychological factors  
- **Target:** Customer shopping preference (Online / In-Store / Hybrid)  
- **License:** CC0 - Public Domain  

### Dataset Download
The dataset can be automatically downloaded using the included script:

```bash
python scripts/download_data.py
```

Make sure you have the Kaggle API configured:
- Install Kaggle CLI: pip install kaggle
- Place your kaggle.json API token at ~/.kaggle/kaggle.json
---
## Setup
1. Clone the repository:

```bash
git clone <repo-url>
cd shopping-ml-project
```
2. Create a virtual environment (recommended):

```bash
conda create -n shopping-ml python=3.11
conda activate shopping-ml
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Download the dataset:

```bash
python scripts/download_data.py
```
---

## Usage
1. Explore data in notebooks/01_eda.ipynb
2. Train and evaluate models in the respective algorithm notebooks
3. Preprocessing and reusable code are in src/
4. Compare model performance in notebooks/09_model_comparison.ipynb (optional)

---
## Learning Objectives

For each algorithm you will:
1. Understand mathematical intuition.
2. Understand bias vs variance trade-off.
3. Know when to use it.
4. Know when not to use it.
5. Tune main hyperparameters.
6. Implement in scikit-learn.
7. Interpret model output.

----

## License

This project is licensed under CC0 - Public Domain.

---

## Author

Omar Gamie – aspiring ML engineer focused on  ML  and  data projects.
