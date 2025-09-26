# Credit Risk Analysis – Model Comparison Report

## 1. Project Goal
The objective of this project is to analyze the **UCI Credit Card Default dataset** and build machine learning models to **predict the probability of default** for the next month.  
This is a classical credit risk problem, where the goal is not only classification accuracy, but also the ability to generate probability scores that support financial decision-making.

---

## 2. Data Preparation & EDA
Before modeling, I conducted **Exploratory Data Analysis (EDA)** and basic cleaning:

- **Distribution analysis**: inspected distributions of numeric features like `AGE`, `LIMIT_BAL`, `BILL_AMT1–6`, `PAY_AMT1–6`.  
- **Outliers analysis**: used boxplots to detect extreme values in bill/payment data.  
- **Correlation analysis**: heatmap showed strong correlation among bill amounts and repayment amounts.  
- **Categorical vs target**: compared default rates across **SEX, EDUCATION, MARRIAGE**.  
- **Trend analysis**: plotted 6-month trends for bills, payments, and repayment status.  

From EDA, I saw that repayment history and utilization ratios are the most informative drivers of default.

---

## 3. Feature Engineering

### Initial Feature Engineering
- Dropped irrelevant column `ID`.  
- Fixed categorical encoding: merged invalid `EDUCATION` and `MARRIAGE` values into “others”.  
- Converted `SEX` into binary (0=male, 1=female).  
- Applied one-hot encoding to `EDUCATION` and `MARRIAGE`.  
- Saved as `processed_data_initial.csv`.

### Advanced Feature Engineering
To enrich the dataset with more business-relevant signals:

- **Ratio features**: credit utilization (bill/limit), payment ratios (payment/bill, payment/limit).  
- **Trend features**: change over 6 months in bill amounts, repayments, and bill-payment gaps.  
- **Aggregation features**: mean, std, and max of bills and payments.  
- **Behavior features**: number of overdue months, longest overdue streak, months with full repayment.  
- **Interaction features**: Age × Limit, Education × Limit.  

Saved as `processed_data_advanced.csv`.

---

## 4. Baseline Models
I started with two **baseline models**:

- **Logistic Regression**: chosen for interpretability and as a standard benchmark in credit scoring.  
- **XGBoost Classifier**: a strong gradient boosting model, commonly used in Kaggle/tabular data problems.  

**Results**:  
- Logistic Regression: Accuracy ~0.78, ROC-AUC ~0.63.  
- XGBoost: Accuracy ~0.815, ROC-AUC ~0.77.  

Conclusion: Logistic Regression failed to capture default patterns, while XGBoost already showed strong predictive power.

---

## 5. Advanced Models (with engineered features)

- **Advanced Logistic Regression**:  
  - Accuracy ~0.78, ROC-AUC improved slightly to ~0.67.  
  - Recall for defaults still extremely low (~0.03).  
  - Linear model too weak even with advanced features.  

- **Advanced XGBoost**:  
  - Accuracy ~0.815, ROC-AUC ~0.778.  
  - Precision/Recall for defaults ~0.65 / 0.36.  
  - Feature engineering didn’t improve much since XGBoost already learns complex interactions.  

---

## 6. Hyperparameter Tuning (XGBoost)
To push performance further, I applied **RandomizedSearchCV** with cross-validation:

- Tuned parameters: learning rate, n_estimators, max_depth, min_child_weight, subsample, colsample_bytree, regularization.  
- Best model achieved:  
  - Accuracy: ~0.819  
  - ROC-AUC: ~0.782  
  - Precision (defaults): ~0.67  
  - Recall (defaults): ~0.36  

**Interpretation**:  
- Tuned XGBoost became the best-performing model.  
- Gains are modest but important in credit risk where every fraction of AUC matters.  
- Most importantly, it provides **default probability scores** for each client.

---

## 7. Additional Models
For robustness, I also tested:

- **Random Forest**: Accuracy ~0.81, ROC-AUC ~0.764.  
- **LightGBM**: Accuracy ~0.815, ROC-AUC ~0.773.  

Both are competitive but did not outperform Tuned XGBoost.  
LightGBM was faster but recall trade-offs were similar.

---

## 8. Final Recommendation
- **Tuned XGBoost** is the final model.  
- It achieved the best ROC-AUC (~0.782) while maintaining accuracy and precision.  
- Generates **default probability scores**, which can be used by financial institutions to:  
  - Flag high-risk customers.  
  - Adjust lending thresholds.  
  - Support credit policy design.

---

## 9. Project Structure
- **data/** → raw and processed datasets (`initial`, `advanced`).  
- **notebooks/** → EDA and feature engineering scripts.  
- **models/** → baseline, advanced, tuned models for Logistic, XGBoost, Random Forest, LightGBM.  
- **reports/** → ROC curve plots and this summary report.  

---

## 10. Key Takeaways
- Feature engineering adds interpretability but tree-based models already capture non-linear patterns.  
- Logistic Regression is too weak for this dataset.  
- XGBoost (tuned) provides the best trade-off between predictive power and business interpretability.  
- Final output: **default probability scores** for use in credit risk management.