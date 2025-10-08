# ====================================
# Data Loading & Cleaning Script
# ====================================

import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path


# ==============================
# 1- Load Data
# ==============================
@st.cache_data
def load_data(file_path):
    """
    Load tabular data into a pandas DataFrame.
    Supports: CSV, Excel, JSON
    """
    if isinstance(file_path, (str, Path)):
        file_name = str(file_path).lower()
    else:  # UploadedFile from Streamlit
        file_name = file_path.name.lower()

    if file_name.endswith(".csv"):
        df = pd.read_csv(file_path)
    elif file_name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(file_path)
    elif file_name.endswith(".json"):
        df = pd.read_json(file_path)
    else:
        raise ValueError(
            "❌ Unsupported File Format. Please upload CSV, Excel, or JSON.")

    st.success(f"✅ Data Loaded Successfully! Shape = {df.shape}")
    return df


# ==============================
# 2- Full Cleaning Pipeline
# ==============================
@st.cache_data
def clean_data(df):
    """
    Run full cleaning pipeline on DataFrame:
    - Handle missing values
    - Remove duplicates
    - Clip outliers
    """
    log = {}
    st.info("Starting data cleaning pipeline...")

    # Missing values
    with st.expander("Handling Missing Values"):
        df = handle_missing(df, log)

    # Duplicates
    with st.expander("Removing Duplicates"):
        df = remove_duplicates(df, log)

    # Outliers
    with st.expander("Clipping Outliers"):
        df = clip_outliers(df, log)

    st.success("✨ Data Cleaning Completed Successfully!")
    st.write("### 📝 Cleaning Summary")
    st.json(log)

    return df


# ==============================
# Handle Missing
# ==============================
def handle_missing(df, log=None):
    """
    Fill missing values:
    - Numeric: mean if not skewed, else median
    - Categorical: mode
    """
    df2 = df.copy()
    num_cols = df2.select_dtypes(include=np.number).columns
    cat_cols = df2.select_dtypes(include="object").columns

    missing_count_before = int(df2.isnull().sum().sum())
    if log is not None:
        log["missing_values_before"] = missing_count_before

    st.write(f"🔍 Total Missing Values before cleaning: {missing_count_before}")

    if missing_count_before > 0:
        # Numeric Columns
        for col in num_cols:
            if df2[col].isnull().any():
                skewness = df2[col].skew()
                method = "median" if abs(skewness) > 1 else "mean"
                st.write(f"➡️ Filling '{col}' with **{method}**")
                if method == "median":
                    df2[col] = df2[col].fillna(df2[col].median())
                else:
                    df2[col] = df2[col].fillna(df2[col].mean())

        # Categorical Columns
        for col in cat_cols:
            if df2[col].isnull().any():
                st.write(f"➡️ Filling '{col}' with **mode**")
                df2[col] = df2[col].fillna(df2[col].mode()[0])

    missing_count_after = int(df2.isnull().sum().sum())
    st.write(f"🔍 Total Missing Values after cleaning: {missing_count_after}")

    if log is not None:
        log["missing_values_after"] = missing_count_after

    return df2


# ==============================
# Remove Duplicates
# ==============================
def remove_duplicates(df, log=None):
    """
    Remove duplicate rows.
    """
    len_before = len(df)
    df2 = df.drop_duplicates()
    len_after = len(df2)
    removed = len_before - len_after

    st.write(f"🗑️ Removed **{removed}** duplicate rows")

    if log is not None:
        log["duplicates_removed"] = removed

    return df2


# ==============================
# Clip Outliers
# ==============================
def clip_outliers(df, log=None):
    """
    Clip outliers in numeric columns using the IQR method.
    """
    df2 = df.copy()
    num_cols = df2.select_dtypes(include=np.number).columns
    outlier_summary = {}

    for col in num_cols:
        Q1 = df2[col].quantile(0.25)
        Q3 = df2[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        outliers = df2[(df2[col] < lower) | (df2[col] > upper)]
        count_outliers = len(outliers)

        if count_outliers > 0:
            df2[col] = df2[col].clip(lower, upper)
            st.write(f"✂️ Clipped **{count_outliers}** outliers in '{col}'")
        else:
            st.write(f"✅ No outliers found in '{col}'")

        outlier_summary[col] = int(count_outliers)

    if log is not None:
        log["outliers"] = outlier_summary

    return df2
