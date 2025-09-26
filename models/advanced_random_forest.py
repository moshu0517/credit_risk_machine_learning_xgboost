#%%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, roc_curve
import matplotlib.pyplot as plt

df = pd.read_csv("/Users/moshu/Desktop/Git-Project/Credit Risk Analysis/data/processed_data_advanced.csv")

#%%
# --- Define features and target ---
X = df.drop(columns=["default payment next month"])
y = df["default payment next month"]

# --- Train-test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Random Forest Classifier ---
rf = RandomForestClassifier(
    n_estimators=200,        # number of trees
    max_depth=None,          # let trees expand fully
    min_samples_split=2,     # default split rule
    min_samples_leaf=1,      # default leaf rule
    random_state=42,
    n_jobs=-1                # use all CPU cores
)
rf.fit(X_train, y_train)

# --- Predictions ---
y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)[:, 1]

# --- Metrics ---
print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# --- ROC Curve ---
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc_score(y_test, y_prob):.2f})")
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Advanced Random Forest")
plt.legend(loc="lower right")
plt.savefig("/Users/moshu/Desktop/Git-Project/Credit Risk Analysis/reports/advanced_random_forest_roc.png")
plt.show()

# --- Random Forest Result ---
# Accuracy: ~0.81, ROC-AUC: ~0.76
# The model is clearly stronger than Logistic Regression, because it does capture non-linear relationships.
# However, recall for the default class (class 1) is still limited (~0.37).
# Precision for class 1 (~0.63) is decent, but the low recall means we still miss many defaults.
# Overall, Random Forest works as a good benchmark, but boosting models (like XGBoost) tend to perform better
# because they reduce bias and capture more complex patterns.

#%%