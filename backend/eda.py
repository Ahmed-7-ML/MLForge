# ===== eda_script.py =====
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# إعدادات الشكل
sns.set(style="whitegrid", palette="muted")
plt.rcParams["figure.figsize"] = (10, 6)

# ==============================
# 1) Load Data
# ==============================
file_name=input('please enter the file name')
df = pd.read_csv(file_name + '.csv')

# ==============================
# 2) Basic Info
# ==============================
print(" Dataset Shape:", df.shape)
print("\n Data Types:\n", df.dtypes)
print("\n Missing Values:\n", df.isnull().sum())
print("\n Summary Stats:\n", df.describe(include="all").T)

# ==============================
# 3) Handle Missing Values (اختياري)
# ==============================
df = df.dropna()  

# ==============================
# 4) Numerical Features
# ==============================
numeric_cols = df.select_dtypes(include=[np.number]).columns

for col in numeric_cols:
    plt.figure()
    sns.histplot(df[col].dropna(), kde=True, bins=30)
    plt.title(f"Distribution of {col}")
    plt.show()

    plt.figure()
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.show()

# ==============================
# 5) Categorical Features
# ==============================
cat_cols = df.select_dtypes(include=["object", "category"]).columns

for col in cat_cols:
    plt.figure()
    sns.countplot(x=df[col], order=df[col].value_counts().index)
    plt.title(f"Countplot of {col}")
    plt.xticks(rotation=45)
    plt.show()

# ==============================
# 6) Correlation Heatmap
# ==============================
if len(numeric_cols) > 1:
    plt.figure(figsize=(12, 8))
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.show()

# ==============================
# 7) Pairplot
# ==============================
if 1 < len(numeric_cols) <= 5:  # لتفادي البطء
    sns.pairplot(df[numeric_cols].dropna())
    plt.show()

print("\n✅ EDA Completed Successfully!")
