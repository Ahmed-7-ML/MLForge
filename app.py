import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from backend.eda import perform_eda
from backend.data import load_data, clean_data


# Global Configurations
st.set_page_config(page_title="EDA Tool", layout="wide")
sns.set_style(style="whitegrid", palette="muted")
plt.rcParams["figure.figsize"] = (10, 6)


# Streamlit App
st.title("📊 EDA & Data Cleaning App")

# Upload Tabular File
uploaded_file = st.file_uploader("Upload your dataset (CSV, Excel, JSON)", type=[
                                "csv", "xlsx", "xls", "json"])

if uploaded_file is not None:
    # Save uploaded file to a temporary location
    file_bytes = uploaded_file.read()
    file_name = uploaded_file.name

    with open(file_name, "wb") as f:
        f.write(file_bytes)

    # After we load file -> load data
    df = load_data(file_name)

    st.subheader("📌 Preview of Data")
    st.dataframe(df.head())

    # Cleaning Data
    st.subheader("🧼 Data Cleaning")
    do_clean = st.checkbox("🧹 Run Data Cleaning Pipeline", value=True)

    if do_clean:
        df = clean_data(df)
        st.success("✅ Data cleaned successfully!")

    # Run EDA
    st.subheader("🔍 Exploratory Data Analysis (EDA)")
    st.info("The following plots and statistics are generated automatically:")

    perform_eda(df)
