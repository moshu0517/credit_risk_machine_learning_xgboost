
import pandas as pd
import numpy as np

df = pd.read_csv('/Users/moshu/Desktop/Git-Project/Supply Chain/supply_chain_data.csv')
print(df.head(10))
print(df.shape)
print(df.columns)

# Complete XGBoost Process for Supply Chain Credit Risk Classification
# Your dataset: 1000 rows × 34 columns

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score, accuracy_score
import xgboost as xgb
from sklearn.multiclass import OneVsRestClassifier
import warnings
warnings.filterwarnings('ignore')


# 综合延迟天数、延迟百分比、违约情况
def create_risk_tiers_v2(row):
    score = 0
    
    # 平均延迟天数评分 (40%权重)
    if row['Avg Payment Delay Days'] > 30: score += 40
    elif row['Avg Payment Delay Days'] > 15: score += 25
    elif row['Avg Payment Delay Days'] > 5: score += 10
    
    # 延迟付款百分比 (30%权重) 
    if row['Delayed Payments Pct'] > 50: score += 30
    elif row['Delayed Payments Pct'] > 25: score += 20
    elif row['Delayed Payments Pct'] > 10: score += 10
    
    # 是否违约过 (20%权重)
    if row['Is Default'] == 1: score += 20
    
    # 最大延迟天数 (10%权重)
    if row['Max Payment Delay Days'] > 60: score += 10
    elif row['Max Payment Delay Days'] > 30: score += 5
    
    # 根据总分分级
    if score >= 70: return 'Very High Risk'
    elif score >= 45: return 'High Risk'  
    elif score >= 20: return 'Medium Risk'
    else: return 'Low Risk'

# 应用方案2给每个客户分配风险等级
df['Credit_Risk_Tier'] = df.apply(create_risk_tiers_v2, axis=1)

# 然后就会看到：
# Low Risk: 250人
# Medium Risk: 300人  
# High Risk: 280人
# Very High Risk: 170人

numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
print(f'numerical columns ({len(numerical_columns)}):')
print(numerical_columns)

categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
print(f'\ncategorical columns ({len(categorical_columns)}):')
print(categorical_columns)

print(f"\n all columns types:")

y = df['Credit_Risk_Tier']
x =df.drop('Credit_Risk_Tier', axis=1)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

for col in categorical_columns:
    if col != 'Credit_Risk_Tier':  # 排除目标变量
        le_col = LabelEncoder()
        x[col] = le_col.fit_transform(x[col])

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state= 42)

model = xgb.XGBClassifier(random_state=42)
model.fit(x_train, y_train)

importance = model.feature_importances_
features_name = x.columns

feature_importance = pd.DataFrame({
    'feature':features_name,
    'importance': importance
}).sort_values('importance',ascending=False)

print(feature_importance)
print(feature_importance.head(10))

y_pred = model.predict(x_test)


# from sklearn.metrics import accuracy_score
# print('accuracy:', accuracy_score(y_test, y_pred))

# from sklearn.metrics import classification_report
# print(classification_report(y_test, y_pred))
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred))

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize

# 把 y_test 和 y_pred_proba 转换成 one-vs-rest 的形式
y_test_bin = label_binarize(y_test, classes=[0,1,2,3])
y_pred_proba = model.predict_proba(x_test)  # 预测概率

auc = roc_auc_score(y_test_bin, y_pred_proba, average="macro", multi_class="ovr")
print("ROC AUC Score:", auc)