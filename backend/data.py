# ===== cleaning_script.py =====
import pandas as pd
import numpy as np

# ==============================
# 1) Load Data
# ==============================   
df = pd.read_csv("shopping_trends.csv")

print("🔹 Before Cleaning:")
print("Shape:", df.shape)
print("Missing Values:\n", df.isnull().sum())
print("Duplicates:", df.duplicated().sum())

# ==============================
# 2) Remove Duplicates
# ==============================
df = df.drop_duplicates()

# ==============================
# 3) Handle Missing Values
# ==============================
# 
df = df.dropna(axis=1, how="all")


df = df.dropna(axis=0, how="all")


num_cols = df.select_dtypes(include=[np.number]).columns
for col in num_cols:
    df[col].fillna(df[col].mean(), inplace=True)


cat_cols = df.select_dtypes(include=["object", "category"]).columns
for col in cat_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].mode()[0], inplace=True)

# ==============================
# 4) Standardize Column Names
# ==============================
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# ==============================
# 5) Encoding 
# ==============================

for col in cat_cols:
    df[col] = df[col].astype("category").cat.codes

# ==============================
# 6) Save Cleaned Data
# ==============================
output_file = "cleaned_dataset.csv"
df.to_csv(output_file, index=False)

print("\n✅ Cleaning Completed Successfully!")
print("🔹 After Cleaning:")
print("Shape:", df.shape)
print("Missing Values:\n", df.isnull().sum())
print("Duplicates:", df.duplicated().sum())
# ===== cleaning_script.py =====
import pandas as pd
import numpy as np

# ==============================
# 1) Load Data
# ==============================
file_path = "your_dataset.csv"   
df = pd.read_csv(file_path)

print("🔹 Before Cleaning:")
print("Shape:", df.shape)
print("Missing Values:\n", df.isnull().sum())
print("Duplicates:", df.duplicated().sum())

# ==============================
# 2) Remove Duplicates
# ==============================
df = df.drop_duplicates()

# ==============================
# 3) Handle Missing Values
# ==============================
df = df.dropna(axis=1, how="all")

df = df.dropna(axis=0, how="all")

num_cols = df.select_dtypes(include=[np.number]).columns
for col in num_cols:
    df[col].fillna(df[col].mean(), inplace=True)


cat_cols = df.select_dtypes(include=["object", "category"]).columns
for col in cat_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].mode()[0], inplace=True)

# ==============================
# 4) Standardize Column Names
# ==============================
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# ==============================
# 5) Encoding (اختياري)
# ==============================

for col in cat_cols:
    df[col] = df[col].astype("category").cat.codes

# ==============================
# 6) Save Cleaned Data
# ==============================
output_file = "cleaned_dataset.csv"
df.to_csv(output_file, index=False)

print("\n Cleaning Completed Successfully!")
print("🔹 After Cleaning:")
print("Shape:", df.shape)
print("Missing Values:\n", df.isnull().sum())
print("Duplicates:", df.duplicated().sum())
print(f"\n Cleaned dataset saved as: {output_file}")

