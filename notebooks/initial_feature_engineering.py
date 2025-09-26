
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('/Users/moshu/Desktop/Git-Project/Credit Risk Analysis/data/processd_default _credit_card_clients.csv',header=1)  #jump the first row as it's the name

# %%
# Data cleaning & processing - Fix column names and data types

# --- Basic cleaning ---
# strip whitespace from column names, drop any weird unnamed columns
df.columns = df.columns.str.strip()
df = df.loc[:, ~df.columns.str.contains(r"^Unnamed", case=False)]

# Quick check: what do my columns and dtypes look like before any casting
print("Columns:", list(df.columns))
print("Dtypes (before):")
print(df.dtypes)

# --- Convert numeric features ---
# I'll just pick the numeric groups by name, no need to overcomplicate
numeric_cols = [
    "LIMIT_BAL","AGE",
    "PAY_0","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6",
    "BILL_AMT1","BILL_AMT2","BILL_AMT3","BILL_AMT4","BILL_AMT5","BILL_AMT6",
    "PAY_AMT1","PAY_AMT2","PAY_AMT3","PAY_AMT4","PAY_AMT5","PAY_AMT6"
]
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

# --- Convert categorical features ---
# SEX, EDUCATION, MARRIAGE are not numbers in the sense of math,
# they are categories, so I’ll mark them as category dtype
categorical_cols = ["SEX","EDUCATION","MARRIAGE"]
df[categorical_cols] = df[categorical_cols].astype("category")

# --- Target column ---
# "default payment next month" is my label (Y), keep it as integer 0/1
# Find the target column by approximate name
target_col = [c for c in df.columns if "default" in c.lower()][0]
print("Target column detected:", target_col)

df[target_col] = pd.to_numeric(df[target_col], errors="coerce").astype(int)
# Quick sanity check after processing
print("Dtypes (after):")
print(df.dtypes)
print(df.head())

# %%
# Step 2: Feature Selection - Remove Irrelevant Features

# --- Must-remove features ---
# ID is just an identifier, not predictive, so drop it
if "ID" in df.columns:
    df = df.drop(columns=["ID"])

# --- Low variance filter ---
# if a column barely varies, it carries no useful information
from sklearn.feature_selection import VarianceThreshold
selector = VarianceThreshold(threshold=0.01)  # drop cols with <1% variance
selector.fit(df.drop(columns=[target_col]))
low_var_cols = df.drop(columns=[target_col]).columns[~selector.get_support()]
print("Low variance columns:", low_var_cols.tolist())

# --- Correlation with target ---
# check which features are weakly related to default
corr = df.corr(numeric_only=True)[target_col].sort_values(ascending=False)
print("Correlation with target:")
print(corr)

# --- Multicollinearity filter ---
# if two features are highly correlated, they are redundant
corr_matrix = df.drop(columns=[target_col]).corr(numeric_only=True).abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
high_corr = [column for column in upper.columns if any(upper[column] > 0.95)]
print("Highly correlated columns (multicollinearity):", high_corr)

# --- Feature selection summary ---
# ID is just an identifier, it has no predictive value, so I will drop it.
# For multicollinearity: features like BILL_AMT1 ~ BILL_AMT6 are naturally correlated,
# but since I'm using tree-based models (like XGBoost), linear dependence is not a problem.
# So I will keep them instead of dropping. ：）


# %%
# Step 3: Handle categorical variables

# This is a public dataset (UCI Credit Card Default).
# According to the official documentation, the categorical variables are encoded as follows:
# SEX:        1 = male, 2 = female
# EDUCATION:  1 = graduate school, 2 = university, 3 = high school, 4 = others
#             values {0, 5, 6} are invalid → we will merge them into "others"
# MARRIAGE:   1 = married, 2 = single, 3 = others
#             value {0} is invalid → we will merge it into "others"

print(df["EDUCATION"].value_counts())
print(df["MARRIAGE"].value_counts())
print(df["SEX"].value_counts())

# --- Merge invalid values into 'others' ---
df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})

# --- Encode categorical variables ---
# SEX has only 2 categories, map to binary 0/1 for clarity
df["SEX"] = df["SEX"].map({1: 0, 2: 1})  # 0=male, 1=female

# EDUCATION and MARRIAGE have multiple categories
# For tree-based models (XGBoost/RandomForest), integer encoding works fine
categorical_cols = ["EDUCATION", "MARRIAGE"]
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

print("Categorical variables have been cleaned and encoded.")


# %%
# %%
# Step 4: Save processed dataset

# Save to data/processed/ folder for clarity
output_path = "/Users/moshu/Desktop/Git-Project/Credit Risk Analysis/data/processed_data_initial.csv"
df.to_csv(output_path, index=False)

print(f"Processed dataset saved to: {output_path}")
# %%
