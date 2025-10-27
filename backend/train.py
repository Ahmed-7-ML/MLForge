import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="Car Price Prediction Dashboard", layout="wide")

st.title("🚗 Car Sales Prediction Dashboard")
st.write("Explore, clean, and model car price data interactively.")

# ---------------- Upload or Load Dataset ----------------
uploaded = st.file_uploader("Upload your car_sales.csv file", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)
else:
    st.info("No file uploaded. Using sample dataset from Kaggle (Car Sales).")
    df = pd.read_csv("1.04.Real-life example.csv")  # ضع ملفك في نفس المجلد

st.subheader("📊 Dataset Overview")
st.write(df.head())

# ---------------- Data Cleaning ----------------
st.subheader("🧹 Data Cleaning")

st.write("Missing values before cleaning:")
st.write(df.isnull().sum())

df = df.dropna(subset=["Price"])  # Remove rows without target
df = df.dropna()  # Simple cleaning for demo

# Remove outliers from Price
q1, q3 = df["Price"].quantile([0.25, 0.75])
iqr = q3 - q1
lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
df = df[(df["Price"] >= lower) & (df["Price"] <= upper)]

st.write("✅ Data cleaned successfully. Remaining shape:", df.shape)

# ---------------- Visualization ----------------
st.subheader("📈 Exploratory Data Analysis")

col1, col2 = st.columns(2)
with col1:
    fig, ax = plt.subplots()
    sns.histplot(df["Price"], bins=30, kde=True, color="skyblue", ax=ax)
    plt.title("Price Distribution")
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x="Year", y="Price", hue="Brand", ax=ax)
    plt.title("Price vs Year by Brand")
    st.pyplot(fig)

# ---------------- Feature & Target ----------------
target = "Price"
features = df.drop(columns=[target])
numeric_features = features.select_dtypes(include=np.number).columns.tolist()
categorical_features = features.select_dtypes(exclude=np.number).columns.tolist()

# ---------------- Data Split ----------------
X = features
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------- Preprocessing ----------------
numeric_transformer = Pipeline([("scaler", StandardScaler())])
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# ---------------- Models ----------------
models = {}

# Linear Regression
models["Linear Regression"] = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LinearRegression())
])

# Polynomial Regression
poly = PolynomialFeatures(degree=2, include_bias=False)
models["Polynomial Regression"] = Pipeline([
    ("preprocessor", preprocessor),
    ("poly", poly),
    ("model", LinearRegression())
])

# Decision Tree
param_tree = {"model__max_depth": [5, 10, 15]}
models["Decision Tree"] = GridSearchCV(
    Pipeline([("preprocessor", preprocessor),
              ("model", DecisionTreeRegressor(random_state=42))]),
    param_grid=param_tree, cv=3
)

# KNN
param_knn = {"model__n_neighbors": [3, 5, 7]}
models["KNN"] = GridSearchCV(
    Pipeline([("preprocessor", preprocessor),
              ("model", KNeighborsRegressor())]),
    param_grid=param_knn, cv=3
)

# Random Forest
param_rf = {"model__n_estimators": [100, 200], "model__max_depth": [10, 20]}
models["Random Forest"] = GridSearchCV(
    Pipeline([("preprocessor", preprocessor),
              ("model", RandomForestRegressor(random_state=42))]),
    param_grid=param_rf, cv=3
)

# ---------------- Training and Evaluation ----------------
results = []

st.subheader("⚙️ Model Training & Evaluation")

for name, model in models.items():
    with st.spinner(f"Training {name}..."):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        results.append({"Model": name, "R²": r2, "MSE": mse, "RMSE": rmse})
        st.success(f"{name} trained successfully ✅")

# ---------------- Results Table ----------------
results_df = pd.DataFrame(results)
st.subheader("📊 Model Performance Metrics")
st.dataframe(results_df)

# ---------------- Plot Comparison ----------------
st.subheader("📉 Model Comparison (R² Score)")
fig, ax = plt.subplots(figsize=(8,4))
sns.barplot(data=results_df, x="Model", y="R²", palette="viridis")
plt.title("Model Comparison (R² Score)")
plt.xticks(rotation=30)
st.pyplot(fig)
