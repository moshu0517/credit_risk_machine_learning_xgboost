#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('/Users/moshu/Desktop/Git-Project/Credit Risk Analysis/data/processd_default _credit_card_clients.csv',header=1)  #jump the first row as it's the name

#%%
# Basic overview
print(df.shape)
print(df.head())
print(df.columns)


# %%
# Data distribution analysis 
# Select only numerical columns
num_cols = df.select_dtypes(include=['int64', 'float64']).columns

# Set up the grid size: 
cols_per_page = 8
total_pages = (len(num_cols) + cols_per_page - 1) // cols_per_page

for page in range(total_pages):
    start = page * cols_per_page
    end = min(start + cols_per_page, len(num_cols))
    sub_cols = num_cols[start:end]
    
    n_rows = (len(sub_cols) + 3) // 4  
    plt.figure(figsize=(20, n_rows * 4))
    
    for i, col in enumerate(sub_cols, 1):
        plt.subplot(n_rows, 4, i)
        sns.histplot(df[col], kde=True, bins=30, color="blue")
        plt.title(col)
    
    plt.tight_layout()
    plt.savefig("distribution.png", dpi=300, bbox_inches='tight')
   # plt.show()
    
# %%
# Outliers analysis 
# Select numerical columns for outlier check
# Define subplot layout (4 columns per row)
n_cols = 4
n_rows = -(-len(num_cols) // n_cols)  # Ceiling division to get enough rows

# Create figure with dynamic size
plt.figure(figsize=(16, n_rows * 4))

# Plot boxplots for each numeric feature
for i, col in enumerate(num_cols, 1):
    plt.subplot(n_rows, n_cols, i)
    sns.boxplot(y=df[col], color="skyblue")
    plt.title(col, fontsize=10)

# Add overall title for the figure
plt.suptitle("Outliers Analysis", fontsize=16, y=1.02)

# Adjust layout to prevent overlapping
plt.tight_layout()

# Save figure to local file 
plt.savefig("outliers_analysis.png", dpi=300, bbox_inches="tight")

# Show the plots
plt.show()
# %%
# Correlation analysis

# Select only numerical columns for correlation
num_cols = df.select_dtypes(include=['int64', 'float64'])

# Compute correlation matrix
corr_matrix = num_cols.corr()

# Plot heatmap
plt.figure(figsize=(14, 10))
sns.heatmap(corr_matrix, cmap="coolwarm", annot=False, cbar=True)
plt.title("Correlation Heatmap", fontsize=16)

# Save the plot
plt.savefig("correlation_analysis.jpeg", dpi=300, bbox_inches='tight')
plt.show()

# %%
# Categorical Variables vs Target

# Define categorical columns to analyze
cat_cols = ['SEX', 'EDUCATION', 'MARRIAGE']

# Create subplots for all categorical variables
fig, axes = plt.subplots(1, len(cat_cols), figsize=(18, 5))
fig.suptitle("Categorical Variables vs Target (Default Payment Next Month)", fontsize=16)

for i, col in enumerate(cat_cols):
    sns.countplot(
        x=col, 
        data=df, 
        hue='default payment next month', 
        palette="Set1", 
        ax=axes[i]
    )
    axes[i].set_title(f"{col} vs Target")
    axes[i].set_ylabel("Count")

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("categorical_vs_target.jpeg", dpi=300, bbox_inches="tight")
plt.show()

# %%
# Box plots for multiple key features 
key_features = ['LIMIT_BAL', 'AGE', 'BILL_AMT1', 'PAY_AMT1']  # pick important ones

for col in key_features:
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='default payment next month', y=col, data=df, palette="Set1")
    plt.title(f"{col} vs Default Payment Next Month")
    plt.xlabel("Default Payment Next Month (0 = No, 1 = Yes)")
    plt.ylabel(col)

    # Save each figure separately with feature name in filename
    plt.savefig(f"boxplot_{col}.jpeg", dpi=300, bbox_inches="tight")
    plt.show()
# %%
# Trend analysis: Time-related features

time_features_bill = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']
time_features_payamt = ['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
time_features_paystatus = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']

# Average bill amount trend
plt.figure(figsize=(10, 6))
df[time_features_bill].mean().plot(marker='o')
plt.title("Trend of Average Bill Amounts (Last 6 Months)")
plt.xlabel("Month")
plt.ylabel("Average Bill Amount")
plt.grid(True)
plt.savefig("trend_bill_amounts.jpeg", dpi=300, bbox_inches="tight")
plt.show()

# Average payment amount trend
plt.figure(figsize=(10, 6))
df[time_features_payamt].mean().plot(marker='o', color="green")
plt.title("Trend of Average Payment Amounts (Last 6 Months)")
plt.xlabel("Month")
plt.ylabel("Average Payment Amount")
plt.grid(True)
plt.savefig("trend_payment_amounts.jpeg", dpi=300, bbox_inches="tight")
plt.show()

# Average payment status trend
plt.figure(figsize=(10, 6))
df[time_features_paystatus].mean().plot(marker='o', color="red")
plt.title("Trend of Average Payment Status (Last 6 Months)")
plt.xlabel("Month")
plt.ylabel("Average Status (0 = on-time, 1+ = delay)")
plt.grid(True)
plt.savefig("trend_payment_status.jpeg", dpi=300, bbox_inches="tight")
plt.show()
# %%
