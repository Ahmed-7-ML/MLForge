# ---------------------------------
# Data Loading & Cleaning Script :-
# Loading data from different formats of Tabular Data (CSV, Excel Sheet Or JSON).
# Then we apply data cleaning(Handle missing values, Drop Duplicates, Normalize Column Names & Column Values, Fix Data Types).
# Finally, We introduce a Report for the user about what done on this data and add the ability to download this report.
# ---------------------------------
import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path
from io import BytesIO
import mlflow

# ---------------------------------
# 1- Load Data
# ---------------------------------
@st.cache_data(show_spinner=False)
def load_data(file_path):
    if isinstance(file_path, (str, Path)):
        file_name = str(file_path).lower()
        open_obj = file_path
    else:
        file_name = file_path.name.lower()
        file_bytes = file_path.read()
        open_obj = BytesIO(file_bytes)

    try:
        if file_name.endswith(".csv"):
            df = pd.read_csv(open_obj)
        elif file_name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(open_obj)
        elif file_name.endswith(".json"):
            df = pd.read_json(open_obj)
        else:
            raise ValueError(
                "Unsupported File Format. Only CSV, Excel, JSON allowed.")
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

    st.success(f"Data loaded successfully! Shape: {df.shape}")
    return df

# ---------------------------------
# 2- Full Cleaning Pipeline
# ---------------------------------
def clean_data(df):
    log = {}
    run_id = None

    with mlflow.start_run(run_name="Data Cleaning Pipeline") as run:
        run_id = run.info.run_id

        with st.expander("1. Normalize Column Names & Values", expanded=True):
            df = normalize_cols(df, log)

        with st.expander("2. Handle Timestamp Columns", expanded=True):
            df = handle_timestamps(df, log)

        with st.expander("3. Handle Missing Values", expanded=True):
            df = handle_missing(df, log)

        with st.expander("4. Remove Duplicates", expanded=True):
            df = remove_duplicates(df, log)

        with st.expander("5. Clip Outliers", expanded=True):
            df = clip_outliers(df, log)


    return df, log, run_id

# ---------------------------------
# Normalize Column Names & Values
# ---------------------------------
def normalize_cols(df, log=None):
    df = df.copy()
    original_cols = df.columns.tolist()

    df.columns = (df.columns
            .str.strip()
            .str.lower()
            .str.replace(r'\s+', '_', regex=True)
            .str.replace(r'[^a-z0-9_]', '', regex=True))

    # Drop ID columns
    id_cols = [col for col in df.columns if 'id' in col]
    if id_cols:
        st.info(f"Dropped ID columns: {', '.join(id_cols)}")
        df.drop(columns=id_cols, inplace=True)
        if log:
            log["dropped_id_columns"] = id_cols

    # Normalize string values
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = (df[col]
            .astype(str)
            .str.strip()
            .str.lower()
            .str.replace(r'\s+', '_', regex=True))
        df[col].replace(['nan', 'none', '<na>'], np.nan, inplace=True)

    if log:
        log["normalized_columns"] = True
    return df

# ---------------------------------
# Handle Timestamp Data
# ---------------------------------
def handle_timestamps(df, log=None):
    df = df.copy()
    timestamp_cols = []

    for col in df.columns:
        if df[col].dtype == 'object':
            temp = pd.to_datetime(df[col], errors='coerce')
            if temp.notna().sum() > 0:  # لو فيه قيم صحيحة
                timestamp_cols.append(col)
                df[f'{col}_year'] = temp.dt.year
                df[f'{col}_month'] = temp.dt.month
                df[f'{col}_day'] = temp.dt.day
                df[f'{col}_hour'] = temp.dt.hour
                df[f'{col}_minute'] = temp.dt.minute
                df[f'{col}_weekday'] = temp.dt.weekday
                df.drop(col, axis=1, inplace=True)
                st.success(
                    f"Converted '{col}' → extracted year, month, day, hour, weekday")

    if log:
        log["timestamp_columns_converted"] = timestamp_cols
    return df

# ---------------------------------
# Handle Missing Values
# ---------------------------------
def handle_missing(df, log=None):
    df = df.copy()
    num_cols = df.select_dtypes(include=np.number).columns
    cat_cols = df.select_dtypes(include="object").columns

    missing_before = int(df.isnull().sum().sum())
    st.write(f"Total Missing Values before: **{missing_before:,}**")

    if log:
        log["missing_before"] = missing_before
    mlflow.log_metric("missing_values_before", missing_before)

    if missing_before == 0:
        st.success("No missing values!")
        return df

    # Numeric
    for col in num_cols:
        if df[col].isnull().any():
            skew = df[col].skew()
            method = "median" if abs(skew) > 1 else "mean"
            val = df[col].median() if method == "median" else df[col].mean()
            df[col].fillna(val, inplace=True)
            st.write(f"Filled '{col}' with {method}: {val:.4f}")

    # Categorical
    for col in cat_cols:
        if df[col].isnull().any():
            mode_val = df[col].mode(dropna=True)
            if not mode_val.empty:
                df[col].fillna(mode_val[0], inplace=True)
            else:
                df[col].ffill(inplace=True)
                df[col].bfill(inplace=True)
                df[col].fillna('unknown', inplace=True)
            st.write(f"Filled '{col}' with mode/ffill/bfill")

    missing_after = int(df.isnull().sum().sum())
    st.write(f"Total Missing Values after: **{missing_after:,}**")

    if missing_after == 0:
        st.success("All missing values filled!")
    else:
        st.warning(
            f"{missing_after} values still missing (full null columns?)")

    if log:
        log["missing_after"] = missing_after
    mlflow.log_metric("missing_values_after", missing_after)

    return df

# ---------------------------------
# Remove Duplicates
# ---------------------------------
def remove_duplicates(df, log=None):
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    removed = before - after
    st.write(f"Removed **{removed:,}** duplicate rows")
    if log:
        log["duplicates_removed"] = removed
    mlflow.log_metric("duplicates_removed", removed)
    return df

# ---------------------------------
# Clip Outliers
# ---------------------------------
def clip_outliers(df, log=None):
    df = df.copy()
    num_cols = df.select_dtypes(include=np.number).columns
    total_clipped = 0

    for col in num_cols:
        if df[col].isnull().all():
            continue
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
        clipped = ((df[col] < lower) | (df[col] > upper)).sum()
        if clipped > 0:
            df[col] = df[col].clip(lower, upper)
            st.write(f"Clipped **{clipped:,}** outliers in '{col}'")
            total_clipped += clipped

    st.write(f"Total outliers clipped: **{total_clipped:,}**")
    if log:
        log["total_outliers_clipped"] = total_clipped
    mlflow.log_metric("total_outliers_clipped", total_clipped)

    return df
