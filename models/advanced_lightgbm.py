#%%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, roc_curve
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV

df = pd.read_csv("/Users/moshu/Desktop/Git-Project/Credit Risk Analysis/data/processed_data_advanced.csv")

#%%
# --- Block 2: Baseline LightGBM (simple setup) ---
# --- Define features and target ---
X = df.drop(columns=["default payment next month"])
y = df["default payment next month"]

# --- Train-test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Train LightGBM Classifier ---
# I keep parameters simple here → Advanced features already improve the signal
# LightGBM is usually more efficient than XGBoost for larger datasets
model = LGBMClassifier(random_state=42, n_estimators=500, learning_rate=0.05)
model.fit(X_train, y_train)

# --- Predictions ---
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# --- Evaluation ---
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)
print(f"Accuracy: {accuracy}")
print(f"ROC-AUC: {roc_auc}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# --- ROC Curve ---
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Advanced LightGBM")
plt.legend(loc="lower right")

# save ROC curve to reports folder
plt.savefig("/Users/moshu/Desktop/Git-Project/Credit Risk Analysis/reports/advanced_lightgbm_rocROC.png")

plt.show()

# --- Advanced LightGBM Results ---
# Accuracy is ~0.815, very similar to Random Forest and slightly below Tuned XGBoost (~0.819).
# ROC-AUC is ~0.773, putting it close to baseline XGBoost and below tuned XGBoost (~0.782).
# Precision for minority class (1 = default) is decent at 0.65, meaning when model predicts default it is usually correct.
# Recall for minority class is low at 0.36, so the model still misses many default cases.
# In general, LightGBM shows strong overall performance with fast training, but its recall trade-off looks similar to Random Forest.
# Compared to XGBoost (especially tuned), LightGBM performs slightly worse in both ROC-AUC and recall.
# Still, it’s a competitive model and worth keeping in the model comparison.

