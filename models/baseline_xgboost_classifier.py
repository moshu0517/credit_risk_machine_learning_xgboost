
#%%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, roc_curve
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

df = pd.read_csv("/Users/moshu/Desktop/Git-Project/Credit Risk Analysis/data/processed_data_initial.csv")
# %%
# --- Define features and target ---
X = df.drop(columns=["default payment next month"])
y = df["default payment next month"]

# --- Train-test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Train XGBoost Classifier ---
xgb_clf = XGBClassifier(
    n_estimators=100,      # number of trees
    max_depth=3,          # tree depth
    learning_rate=0.1,    # step size shrinkage
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42
)
xgb_clf.fit(X_train, y_train)

# --- Evaluate baseline performance ---
y_pred = xgb_clf.predict(X_test)
y_proba = xgb_clf.predict_proba(X_test)[:, 1]  # probability of default

print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# --- ROC curve visualization ---
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = roc_auc_score(y_test, y_proba)

plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve - XGBoost Baseline")
plt.legend(loc="lower right")
plt.savefig("/Users/moshu/Desktop/Git-Project/Credit Risk Analysis/reports/basedline_xgboost_roc.png")
plt.show()

# --- Result Interpretation ---
# Accuracy: ~82%, better than Logistic Regression (~78%).
# ROC-AUC: ~0.78, which shows stronger discriminative power compared to Logistic Regression (~0.64).
# Class 0 (non-default): precision 0.84, recall 0.95, very solid performance.
# Class 1 (default): precision ~0.67, recall ~0.36, the model is finally catching some default cases.
# Recall is still low, but at least it's not completely missing them like Logistic Regression did.
# Overall: XGBoost baseline clearly shows stronger non-linear fitting ability than a linear model,
# making it a better starting point. Next step is to improve further through Advanced Feature Engineering + Hyperparameter Tuning.
# %%
