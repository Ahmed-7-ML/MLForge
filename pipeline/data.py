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
@st.cache_data
def load_data(file_path):
    """
    Load tabular data into a pandas DataFrame.
    Supports: CSV, Excel, JSON
    Accepts:
        - str / Path -> path on disk
        - streamlit UploadedFile
    Returns:
        - Pandas Data Frame
    """
    # Determine file_name and bytes
    if isinstance(file_path, (str, Path)):
        file_name = str(file_path).lower()
        open_obj = file_path

    else:  # Streamlit UploadedFile
        file_name = file_path.name.lower()
        # read content into BytesIO so pandas can consume it multiple times
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
            raise ValueError("❌ Unsupported File Format. Please upload CSV, Excel, or JSON.")
    except Exception as e:
        st.error(f"Error reading file: {e}")
        raise

    return df

# ---------------------------------
# 2- Full Cleaning Pipeline
# ---------------------------------
def clean_data(df):
    """
    Run full cleaning pipeline on DataFrame:
    - Normalize column names/values
    - Fix data types (e.g., detect and convert datetime)
    - Handle missing values
    - Remove duplicates
    - Clip outliers
    - Scale/normalize numeric columns
    Returns: cleaned_df, log_dict (UI handled outside)
    """
    log = {}

    # Start MLflow run for logging
    with mlflow.start_run(run_name="Data Cleaning Pipeline") as run:
        with st.expander("Normalize Column Names and Values"):
            df = normalize_cols(df, log)
        # Timestamp Data Handling
        with st.expander("Handle Timestamp Data"):
            df = handle_timestamps(df, log)
        # Missing values
        with st.expander("Handling Missing Values"):
            df = handle_missing(df, log)
        # Duplicates Removal
        with st.expander("Removing Duplicates"):
            df = remove_duplicates(df, log)
        # Outliers Handling
        with st.expander("Clipping Outliers"):
            df = clip_outliers(df, log)


    return df, log, run.info.run_id

# ---------------------------------
# Column Names and Values Normalization
# ---------------------------------
def normalize_cols(df, log=None):
    """
    Normalize Column Names and Values to be in lower case and replace spaces with underscores
    Return a Cleaned Data Frame
    """
    df = df.copy()
    df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_', regex=False)
    # Remove columns containing 'id' (case insensitive)
    cols_to_drop = [col for col in df.columns if 'id' in col]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
    for col in df.select_dtypes(include=['object']).columns:
        try:
            df[col] = df[col].astype(str).str.strip().str.replace(' ', '_', regex=False).str.lower()
            # restore nan where original was nan
            mask_nan = df[col].isin(['nan', 'none', 'none-type'])
            df.loc[mask_nan, col] = np.nan
        except Exception:
            # fallback: skip column if can't be processed
            continue
    return df

# ---------------------------------
# Handle Missing
# ---------------------------------
def handle_missing(df, log=None):
    """
    Fill missing values:
    - Numeric: mean if not skewed, else median
    - Categorical: mode (fallback to empty string if no mode)
    """
    df2 = df.copy()
    num_cols = df2.select_dtypes(include=np.number).columns
    cat_cols = df2.select_dtypes(include="object").columns

    missing_count_before = int(df2.isnull().sum().sum())
    st.write(f"🔍 Total Missing Values before cleaning: {missing_count_before}")

    if log is not None:
        log["missing_values_before"] = missing_count_before
    mlflow.log_metric("missing_values_before", missing_count_before)
    if missing_count_before == 0:
        st.success("No missing values found!")
        return df2

    if missing_count_before > 0:
        # Numeric Columns
        for col in num_cols:
            if df2[col].isnull().any():
                # if column all null skip
                if df2[col].dropna().empty:
                    continue
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
                try:
                    mode_val = df2[col].mode(dropna=True)
                    if not mode_val.empty:
                        fill = mode_val[0]
                        df2[col] = df2[col].fillna(fill)
                    else:
                        df2[col] = df2[col].ffill().bfill() 
                except Exception:
                    df2[col] = df2[col].ffill().bfill()

    missing_count_after = int(df2.isnull().sum().sum())
    st.write(f"🔍 Total Missing Values after cleaning: {missing_count_after}")

    if missing_count_after == 0:
        st.success("All missing values handled successfully!")
    else:
        st.warning(f"Still {missing_count_after} missing values remain (probably entire columns were null)")
    
    if log is not None:
        log["missing_values_after"] = missing_count_after
    mlflow.log_metric("missing_values_after", missing_count_after)

    return df2

# ---------------------------------
# Remove Duplicates
# ---------------------------------
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
    mlflow.log_param("duplicates_removed", removed)

    return df2

# ---------------------------------
# Clip Outliers
# ---------------------------------
def clip_outliers(df, log=None):
    """
    Clip outliers in numeric columns using the IQR method.
    """
    df2 = df.copy()
    num_cols = df2.select_dtypes(include=np.number).columns
    outlier_summary = {}

    for col in num_cols:
        col_non_na = df2[col].dropna()
        if col_non_na.empty:
            outlier_summary[col] = 0
            st.write(f"⚪ Column '{col}' empty or all NaN — skipped outlier clipping.")
            continue

        Q1 = col_non_na.quantile(0.25)
        Q3 = col_non_na.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        outliers_mask = (df2[col] < lower) | (df2[col] > upper)
        outliers_mask = outliers_mask.fillna(False)
        count_outliers = int(outliers_mask.sum())

        if count_outliers > 0:
            df2[col] = df2[col].clip(lower, upper)
            st.write(f"✂️ Clipped **{count_outliers}** outliers in '{col}'")
        else:
            st.write(f"✅ No outliers found in '{col}'")

        outlier_summary[col] = count_outliers

    if log is not None:
        log["outliers"] = outlier_summary
    mlflow.log_param("outliers_clipped", str(outlier_summary))

    return df2

# ---------------------------------
# Handle Timestamp Data
# ---------------------------------
def handle_timestamps(df, log=None):
    """
        - Convert Timestamp Columns into DataTime Columns.
        - Extract Important Features from them.
        - Drop the original columns.
    """
    for col in df.select_dtypes(include=['object']).columns:
        try:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            if not df[col].isnull().all():
                df[f'{col}_year'] = df[col].dt.year
                df[f'{col}_month'] = df[col].dt.month
                df[f'{col}_day'] = df[col].dt.day
                df[f'{col}_hour'] = df[col].dt.hour
                df[f'{col}_minute'] = df[col].dt.minute
                df[f'{col}_second'] = df[col].dt.second
                df[f'{col}_weekday'] = df[col].dt.weekday
                # Drop the original columns to avoid being encdoed
                df.drop(col, axis=1, inplace=True)
                st.success(f"Timestamp column '{col}' converted and features extracted successfully!")
        except:
            pass

    return df
