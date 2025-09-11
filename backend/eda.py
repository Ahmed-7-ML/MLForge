import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from backend.data import load_data, clean_data

def perform_eda(file_path):
    """
    Perform Exploratory Data Analysis (EDA) on the dataset.
    """

    # Load and clean data
    df = load_data(file_path)
    df = clean_data(df)

    # Display basic information
    print("DataFrame Info:")
    print(df.info())
    print("\nDataFrame Statistics Description:")
    print(df.describe(include='all'))

    # Visualizations
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
    plt.title('Missing Values Heatmap')
    plt.show()

    num_cols = df.select_dtypes(include='number').columns

    # Draw distributions and boxplots for numeric columns
    for col in num_cols:
        plt.figure(figsize=(10, 4))
        sns.histplot(df[col], kde=True)
        plt.title(f'Distribution of {col}')
        plt.show()

        plt.figure(figsize=(10, 4))
        sns.boxplot(x=df[col])
        plt.title(f'Boxplot of {col}')
        plt.show()
    
    # Correlation heatmap for numeric columns
    if not num_cols.empty:
        plt.figure(figsize=(8, 6))
        sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm")
        plt.title("Correlation Heatmap")
        plt.show()

    # Countplots for categorical columns
    for col in df.select_dtypes(include='object').columns:
        plt.figure(figsize=(10, 4))
        sns.countplot(y=col, data=df, order=df[col].value_counts().index)
        plt.title(f'Countplot of {col}')
        plt.show()
