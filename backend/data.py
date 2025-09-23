import pandas as pd
import numpy as np
# 1- Loading the Data
def load_data(file_path):
    """
    Load data from a file into a pandas DataFrame.
    Supports CSV, Excel, JSON.
    """
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(file_path)
    elif file_path.endswith('.json'):
        df = pd.read_json(file_path)
    else:
        raise ValueError("Unsupported File Format. Use CSV, Excel, or JSON.")
    return df


# 2- Full Cleaning Pipeline
def clean_data(df):
    """
    Run Full Data Cleaning & Processing Pipeline on the DataFrame.
    """
    df = handle_missing(df)
    df = remove_duplicates(df)
    df = clip_outliers(df)
    df = normalize_strings(df)
    df = convert_dtypes(df)
    df = drop_irrelevant(df)
    df = handle_rare_categories(df)
    df = handle_skewness(df)
    return df


# --- Cleaning Functions ---

def handle_missing(df):
    """
    Fill missing values:
    - Numeric: mean if not skewed, else median
    - Categorical: mode
    - Datetime: forward fill
    - Boolean: fill with mode
    """
    df2 = df.copy()
    num_cols = df2.select_dtypes(include=np.number).columns
    cat_cols = df2.select_dtypes(include='object').columns
    date_cols = df2.select_dtypes(include='datetime').columns
    bool_cols = df2.select_dtypes(include='bool').columns

    print(f"Total Missing Values: {df2.isnull().sum().sum()}")

    # Numeric
    for col in num_cols:
        if df2[col].isnull().any():
            skewness = df2[col].skew()
            if abs(skewness) > 1:
                df2[col] = df2[col].fillna(df2[col].median())
            else:
                df2[col] = df2[col].fillna(df2[col].mean())

    # Categorical
    for col in cat_cols:
        if df2[col].isnull().any():
            df2[col] = df2[col].fillna(df2[col].mode()[0])

    # Datetime
    for col in date_cols:
        if df2[col].isnull().any():
            df2[col] = df2[col].fillna(method='ffill')

    # Boolean
    for col in bool_cols:
        if df2[col].isnull().any():
            df2[col] = df2[col].fillna(df2[col].mode()[0])

    return df2


def remove_duplicates(df):
    """Remove duplicate rows from the DataFrame."""
    if df.duplicated().sum() == 0:
        print("No duplicate rows found.")
        return df
    len_before = len(df)
    df = df.drop_duplicates()
    len_after = len(df)
    print(f"Removed {len_before - len_after} duplicate rows.")
    return df


def clip_outliers(df):
    """Clip outliers in numeric columns using IQR method."""
    df2 = df.copy()
    num_cols = df2.select_dtypes(include=np.number).columns
    for col in num_cols:
        Q1, Q3 = df2[col].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        outliers = df2[(df2[col] < lower) | (df2[col] > upper)]
        if len(outliers) > 0:
            df2[col] = df2[col].clip(lower, upper)
            print(f"Clipped {len(outliers)} outliers in '{col}'")
    return df2


def normalize_strings(df):
    """Trim spaces & lowercase all categorical strings."""
    df2 = df.copy()
    cat_cols = df2.select_dtypes(include='object').columns
    for col in cat_cols:
        df2[col] = df2[col].astype(str).str.strip().str.lower()
    return df2


def convert_dtypes(df):
    """Convert columns to correct data types (numeric, datetime)."""
    df2 = df.copy()
    for col in df2.columns:
        if df2[col].dtype == 'object':
            # Try numeric conversion
            try:
                df2[col] = pd.to_numeric(df2[col])
                continue
            except:
                pass
            # Try datetime conversion
            try:
                df2[col] = pd.to_datetime(df2[col])
            except:
                pass
    return df2


def drop_irrelevant(df, threshold=0.7):
    """Drop columns with more than 70% missing values."""
    df2 = df.copy()
    missing_ratio = df2.isnull().mean()
    to_drop = missing_ratio[missing_ratio > threshold].index
    df2 = df2.drop(columns=to_drop)
    if len(to_drop) > 0:
        print(f"Dropped columns with > {threshold*100}% missing: {list(to_drop)}")
    return df2


def handle_rare_categories(df, threshold=0.01):
    """Group rare categories into 'other'."""
    df2 = df.copy()
    cat_cols = df2.select_dtypes(include='object').columns
    for col in cat_cols:
        freqs = df2[col].value_counts(normalize=True)
        rare = freqs[freqs < threshold].index
        if len(rare) > 0:
            df2[col] = df2[col].replace(rare, 'other')
            print(f"Replaced rare categories in '{col}' with 'other'")
    return df2


def handle_skewness(df):
    """Apply log transform to highly skewed numeric columns."""
    df2 = df.copy()
    num_cols = df2.select_dtypes(include=np.number).columns
    for col in num_cols:
        skewness = df2[col].skew()
        if abs(skewness) > 2:
            df2[col] = np.log1p(df2[col])  # log(1+x) to handle zeros
            print(f"Applied log transform to '{col}' (skew={skewness:.2f})")
    return df2
