# Hyperparameter tuning is like searching for the best balance between underfitting and overfitting.
# I’ll start from the most important knobs (learning_rate, n_estimators), then move on to model complexity (max_depth, min_child_weight), add randomness (subsample, colsample_bytree), regularization (reg_alpha, reg_lambda, gamma), and finally handle class imbalance (scale_pos_weight).
# Step by step, narrowing down the search space, until I get a model that generalizes well on unseen data.

#%%
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, roc_curve
from xgboost import XGBClassifier
import numpy as np

df = pd.read_csv("/Users/moshu/Desktop/Git-Project/Credit Risk Analysis/data/processed_data_advanced.csv")

#%%
# --- Define features and target ---
X = df.drop(columns=["default payment next month"])
y = df["default payment next month"]

# --- Train-test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Baseline model for comparison ---
# Start with a simple XGBoost classifier
xgb = XGBClassifier(
    objective="binary:logistic",
    eval_metric="auc",       # use auc to avoid warning
    use_label_encoder=False,
    random_state=42
)

# --- Hyperparameter tuning with RandomizedSearchCV ---
# I'll search across a wide but reasonable range of parameters
param_dist = {
    "n_estimators": [200, 400, 600, 800],
    "max_depth": [3, 4, 5, 6, 8],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "min_child_weight": [1, 3, 5, 7],
    "gamma": [0, 0.1, 0.2, 0.3],
    "reg_alpha": [0, 0.01, 0.1],
    "reg_lambda": [1, 1.5, 2]
}

# Use AUC as scoring metric
random_search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_dist,
    n_iter=20,                  # number of random combos to try
    scoring="roc_auc",
    cv=3,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train, y_train)

# --- Best model ---
best_model = random_search.best_estimator_
print("Best parameters:", random_search.best_params_)

# --- Evaluate tuned model ---
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# --- ROC Curve ---
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc_score(y_test, y_proba):.2f})")
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Tuned XGBoost")
plt.legend(loc="lower right")
plt.show()

"""
Hyperparameter Tuning (GridSearchCV)
-----------------------------------

- The idea is simple: we have many possible parameter choices, so GridSearchCV 
  just tries every combination like a brute-force search.

- For each candidate, it runs cross-validation (here 3-fold CV) to get a stable score. 
  That’s why you see "Fitting 3 folds for each of 20 candidates, totalling 60 fits".

- Finally, it reports the best parameter set and re-trains the model with it.

- In short: it’s a systematic way to find better hyperparameters instead of guessing.
"""
# %%