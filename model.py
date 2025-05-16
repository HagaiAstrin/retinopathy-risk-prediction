"""
URgent Project – Retinopathy Risk Prediction

This script trains a classification model to identify urgent diabetic retinopathy cases
based on structured electronic health record (EHR) data.

Target classes:
    0 – Normal findings
    1 – Clinically significant findings
    2 – Urgent clinical findings

Data preprocessing steps:
- Use only the latest record per patient
- Remove sparse or irrelevant features
- Handle missing values appropriately
- Address class imbalance via resampling

Model:
- RandomForestClassifier
- Hyperparameter tuning with GridSearchCV (scoring='recall_macro')

Outputs:
- Evaluation metrics
- Confusion matrix heatmap
- Feature importance plot
- SHAP summary plot (class 2)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)
from sklearn.inspection import permutation_importance
from imblearn.over_sampling import SMOTE

# Load dataset
df = pd.read_csv(DATA_PATH)
df.drop(columns=["DIABETES"], inplace=True)

# Define features and labels
target_col = "Findings"
X = df.drop(columns=[target_col])
y = df[target_col]

# Handle class imbalance
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, stratify=y_resampled, random_state=42
)

# Define hyperparameter grid
param_grid = {
    'n_estimators': [520, 530, 540],
    'max_depth': [None, 10, 6],
    'min_samples_split': [5, 10],
    'min_samples_leaf': [4, 10],
    'max_features': ['sqrt', 'log2', None],
    'criterion': ['gini', 'entropy'],
    'class_weight': ['balanced'],
    'max_samples': [None, 0.8, 0.5]
}


# Train model
clf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(clf, param_grid, scoring='recall_macro', cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_

# Predictions
y_pred = best_model.predict(X_test)

# Evaluation
acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# Print basic metrics
print("Best Parameters:", grid_search.best_params_)
print(f"Test Accuracy: {acc:.4f}")
print(report)

# Confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "confusion_matrix.png"))
plt.clf()

# Permutation feature importance
perm_result = permutation_importance(best_model, X_test, y_test, n_repeats=10, random_state=42)
importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": perm_result.importances_mean
}).sort_values(by="Importance", ascending=False)

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=importance_df)
plt.title("Feature Importance (Permutation)")
plt.tight_layout()
plt.clf()

# SHAP Explainability for Class 2
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values[2], X_test, feature_names=X.columns, show=False)
plt.tight_layout()
plt.clf()
