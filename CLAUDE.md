# Artificial Intelligence / Machine Learning Optimization Project

## Repository Origin
Cloned from: https://github.com/Carlos-Javier10/inteligencia_artificial
Author: Carlos-Javier (November 2024)

## Project Purpose
A collection of machine learning scripts covering classification and regression, organized by class session (clase). The goal is to analyze these scripts, identify performance gaps, and optimize models by reintroducing interaction terms and leveraging correlation analysis to improve variable selection.

## Repository Structure

```
repo/
├── clase 5/                          # Class 5: Logistic Regression
│   ├── regresion logistica.py        # Logistic Regression (INCOMPLETE - syntax errors)
│   ├── Social_Network_Ads.csv        # Dataset: social network ad purchases
│   └── Logistic Regression Intuition_06Nov2024.pdf
│
├── clase 6 KNN/                      # Class 6: KNN & SVM
│   ├── regresion logistica.py        # K-Nearest Neighbors (k=5, Minkowski/Euclidean)
│   ├── regresion logistica con SVN.py  # SVM with LINEAR kernel
│   ├── regresion lineal con SVN con kernel rbf.py  # SVM with RBF kernel
│   └── Social_Network_Ads.csv
│
├── clase 7/                          # Class 7: Trees & Ensembles
│   ├── arboles de desicion y random forest.py  # Decision Tree + Random Forest REGRESSION (wine data)
│   ├── ramdom_fores_clasiffier.py    # Random Forest CLASSIFICATION (social network data)
│   ├── regresion logistica con SVN.py  # Decision Tree CLASSIFICATION
│   └── Social_Network_Ads.csv
│
├── regresion/                        # Regression module
│   └── Multiple_Linear_Regression.py # Multiple regression + polynomial + backward elimination
│
├── regresion lineal multiple/        # Multiple Linear Regression module
│   └── Multiple_Linear_Regression.py # Multiple regression + backward elimination (50_Startups)
│
└── regresion lineal simple/          # Simple Linear Regression module
    └── Multiple_Linear_Regression.py # Actually: multiple regression with backward elimination
```

## Datasets Used
- **Social_Network_Ads.csv**: User ID, Gender, Age, EstimatedSalary, Purchased (binary classification)
- **50_Startups.csv**: R&D Spend, Administration, Marketing Spend, State, Profit (regression)
- **Position_Salaries.csv**: Position, Level, Salary (regression)
- **winequality-red.csv**: Wine quality features including alcohol, fixed acidity, etc. (regression)

## Key Findings / Issues to Address
1. **Incomplete code**: clase 5/regresion logistica.py has syntax errors (line 31: `[2.3]` should be `[2,3]`, line 67-69 incomplete)
2. **Only 2 features used**: Classification scripts only use Age + EstimatedSalary, ignoring Gender
3. **No interaction terms**: No scripts create interaction features (e.g., Age * Salary)
4. **No correlation matrix**: No script examines variable correlations before feature selection
5. **Backward elimination is manual**: OLS backward elimination is hardcoded, not automated
6. **No cross-validation**: All scripts use a single train/test split
7. **No hyperparameter tuning**: All models use default or arbitrary hyperparameters
8. **StandardScaler applied to tree models**: Trees don't need feature scaling

## Algorithms Covered
- Logistic Regression, KNN, SVM (linear + RBF), Decision Trees, Random Forest
- Linear Regression (simple + multiple), Polynomial Regression
- Backward Elimination (manual, statsmodels OLS)

## Language Note
All original code comments are in Spanish. Translations are applied when working on the code.
