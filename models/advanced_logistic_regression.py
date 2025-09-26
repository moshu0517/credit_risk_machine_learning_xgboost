
#%%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, roc_curve
import matplotlib.pyplot as plt

df = pd.read_csv("/Users/moshu/Desktop/Git-Project/Credit Risk Analysis/data/processed_data_advanced.csv")

# %%
# --- Define features and target ---
# target column is "default payment next month"
X = df.drop(columns=["default payment next month"])
y = df["default payment next month"]

# --- Train-test split ---
# I’ll keep stratify=y so train/test has the same default ratio
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Train Logistic Regression ---
# liblinear works fine for small/medium dataset
log_reg = LogisticRegression(max_iter=2000, solver="liblinear")
log_reg.fit(X_train, y_train)

# --- Evaluate model performance ---
y_pred = log_reg.predict(X_test)
y_proba = log_reg.predict_proba(X_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# --- Plot ROC Curve ---
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = roc_auc_score(y_test, y_proba)

plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Advanced Logistic Regression")
plt.legend(loc="lower right")
plt.savefig("/Users/moshu/Desktop/Git-Project/Credit Risk Analysis/reports/advanced_logistic_roc.png")
plt.show()

# --- Result Interpretation ---
# Comparing baseline vs advanced logistic regression:
# 1) Accuracy stayed ~77.8% in both cases, which is not surprising because the data is highly imbalanced
#    and the model mainly predicts "no default".
# 2) ROC-AUC improved from ~0.63 → ~0.67 after adding advanced features (ratios, trends, behavioral, aggregation).
#    This means the new features gave the model a bit more signal to separate default vs non-default.
# 3) For Class 1 (default): in baseline the precision and recall were both 0, which means the model never detected defaults.
#    In advanced, precision is ~0.44 but recall is only ~0.03 → so it can finally pick up a few default cases,
#    but it still misses the majority of them.
# 4) Overall: Advanced features do help, but logistic regression as a linear model is too weak
#    to capture the complex default patterns. This is exactly why we need non-linear models like XGBoost or LightGBM next.
# %%
