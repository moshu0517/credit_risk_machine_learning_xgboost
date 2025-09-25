
## EDA Analysis Report
My exploratory data analysis highlights key patterns in credit card default behavior. We observed target imbalance, strong correlations with repayment history, and significant outliers in financial variables. These insights guide feature engineering and model selection.

- **Imbalanced target**: ~22% default vs 78% non-default.
 ![Target Distribution](notebooks/distribution.png)
- **Education & marriage**: Lower education groups show higher default; both married and single groups have notable default rates.
  ![Categorical Variables vs Target](notebooks/categorical_vs_target.jpeg)
- **Repayment history**: PAY_0 to PAY_6 strongly correlate with default status.
  ![Correlation Heatmap](notebooks/correlation_analysis.jpeg)
- **Credit limit & bills**: Defaulters tend to have lower credit limits; bill/payment amounts are highly skewed with extreme outliers.
  ![LIMIT_BAL vs Default](notebooks/boxplot_LIMIT_BAL.jpeg)
- **Age**: Limited predictive power, with median in early 30s.
  ![AGE vs Default](notebooks/boxplot_AGE.jpeg)
