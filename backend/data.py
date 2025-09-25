# Main Data Loading & Data Cleaning Script

# Import Libraries
import pandas as pd
import numpy as np

# 1- Loading the Data
def load_data(file_path):
    """
    Load data from a file into a pandas DataFrame.
    (CSV, Excel, JSON) -> Only Tabular Data.
    """
    file_name = file_path.name.lower()

    if file_name.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_name.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(file_path)
    elif file_name.endswith('.json'):
        df = pd.read_json(file_path)
    else:
        raise ValueError(
            'Unsupported File Format, Please use CSV, Excel or JSON Format.')

    return df

# 2- Data Cleaning
def clean_data(df):
    """
    Run Full Data Cleaning Pipeline on the DataFrame.
    """

    df = handle_missing(df)
    df = remove_duplicates(df)
    df = clip_outliers(df)

    return df

def handle_missing(df):
    """
    Fill missing values:
    - Numeric: mean if not skewed, else median
    - Categorical: mode
    """

    df2 = df.copy()
    num_cols = df2.select_dtypes(include=np.number).columns
    cat_cols = df2.select_dtypes(include='object').columns

    print(f"DataFrame Has {df2.isnull().sum().sum()} Missing Values")

    # Numeric Columns
    for col in num_cols:
        if df2[col].isnull().any():
            skewness = df2[col].skew()
            print(f"Column '{col}' Skewness: {skewness}")
            print(f"Filling missing values in numeric column '{col}'")
            if abs(skewness) > 1:
                df2[col] = df2[col].fillna(df2[col].median())
            else:
                df2[col] = df2[col].fillna(df2[col].mean())

    # Categorical Columns
    for col in cat_cols:
        if df2[col].isnull().any():
            print(f"Filling missing values in categorical column '{col}'")
            df2[col] = df2[col].fillna(df2[col].mode()[0])

    return df2

def remove_duplicates(df):
    """
    Remove duplicate rows from the DataFrame.
    """
    if df.duplicated().sum() == 0:
        print("No duplicate rows found.")
        return df
    
    len_before = len(df)
    df = df.drop_duplicates()
    len_after = len(df)
    print(f"Removed {len_before - len_after} duplicate rows.")
    return df

def clip_outliers(df):
    """
    Clip outliers in numeric columns using the IQR method.
    """

    df2 = df.copy()
    num_cols = df2.select_dtypes(include=np.number).columns
    for col in num_cols:
        Q1 = df2[col].quantile(0.25)
        Q3 = df2[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df2[(df2[col] < lower_bound) | (df2[col] > upper_bound)]
        
        print(f"Column '{col}': has {len(outliers)} outliers")
        if len(outliers) == 0:
            continue
        else:
            df2[col] = df2[col].clip(lower_bound, upper_bound)
            print(f"Clipped outliers in column '{col}' using IQR method.")

    return df2
