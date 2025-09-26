#%%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, roc_curve
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

df = pd.read_csv("/Users/moshu/Desktop/Git-Project/Credit Risk Analysis/data/processed_data_advanced.csv")
# %%
# --- Define features and target ---
# target column is still "default payment next month"
X = df.drop(columns=["default payment next month"])
y = df["default payment next month"]

# --- Train-test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# I'll keep this as a simple baseline first, then later we can tune parameters
xgb_clf = XGBClassifier(
    n_estimators=200,       # number of trees
    max_depth=4,            # depth of each tree
    learning_rate=0.1,      # step size shrinkage
    subsample=0.8,          # subsample ratio for training
    colsample_bytree=0.8,   # subsample ratio for features
    random_state=42,
    use_label_encoder=False,
    eval_metric="logloss"
)
xgb_clf.fit(X_train, y_train)

# --- Evaluate performance ---
y_pred = xgb_clf.predict(X_test)
y_proba = xgb_clf.predict_proba(X_test)[:, 1]  # probability of default

print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# --- ROC Curve visualization ---
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = roc_auc_score(y_test, y_proba)

plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Advanced XGBoost Classifier")
plt.legend(loc="lower right")

# save ROC curve to reports folder
plt.savefig("/Users/moshu/Desktop/Git-Project/Credit Risk Analysis/reports/xgboost_roc_curve.png")

plt.show()


# --- Result Interpretation ---
# Compared to baseline XGBoost, the advanced feature set does not give a big lift.
# Accuracy stays ~81.5%, ROC-AUC ~0.78 (almost the same).
# For Class 1 (defaults), precision and recall are roughly the same (~0.65 precision, ~0.36 recall).
# This tells me:
# 1. Baseline XGBoost was already capturing most of the useful signals.
# 2. The advanced ratio/trend/behavior features might overlap with what XGBoost learns internally.
# 3. To get improvements, I’ll likely need parameter tuning (depth, learning rate, regularization).
# %%

# --- Predict Probabilities (Tuned XGBoost) ---
# predict_proba returns two columns: [P(y=0), P(y=1)]
y_pred_proba = xgb_clf.predict_proba(X_test)[:, 1]
print("First 10 predicted default probabilities:", y_pred_proba[:10])
# %%
