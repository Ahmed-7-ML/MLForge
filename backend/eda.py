# Main EDA Script File
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid", palette="muted")
plt.rcParams["figure.figsize"] = (10, 6)


def perform_eda(df, dropna=False):
    """
    Perform full Exploratory Data Analysis (EDA) on a given DataFrame.
    Includes:
        - Info, Missing values, Summary stats
        - Numeric and categorical distributions
        - Correlation heatmap, Pairplot
        - Boxplots, Violin plots, Swarm plots
        - Scatter plots, Line plots
    """

    # ==============================
    # 1) Basic Info
    # ==============================
    print("\n--- Dataset Shape:", df.shape)
    print("\n--- Data Types:\n", df.dtypes)
    print("\n--- Missing Values:\n", df.isnull().sum())
    print("\n--- Summary Statistics:\n", df.describe(include="all").T)

    # ==============================
    # 2) Split Columns
    # ==============================
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    datetime_cols = df.select_dtypes(include="datetime").columns

    # ==============================
    # 3) Numeric Features
    # ==============================
    for col in numeric_cols:
        plt.figure()
        sns.histplot(df[col].dropna(), kde=True, bins=30, color="steelblue")
        plt.title(f"Distribution of {col}")
        plt.show()

        plt.figure()
        sns.boxplot(x=df[col], color="skyblue")
        plt.title(f"Boxplot of {col}")
        plt.show()

    # ==============================
    # 4) Categorical Features
    # ==============================
    for col in cat_cols:
        plt.figure()
        sns.countplot(x=df[col], order=df[col].value_counts().index, palette="viridis")
        plt.title(f"Countplot of {col}")
        plt.xticks(rotation=45)
        plt.show()

    # ==============================
    # 5) Correlation Heatmap
    # ==============================
    if len(numeric_cols) > 1:
        plt.figure(figsize=(12, 8))
        corr = df[numeric_cols].corr()
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Heatmap")
        plt.show()

    # ==============================
    # 6) Pairplot
    # ==============================
    if 1 < len(numeric_cols) <= 6 and len(df) < 2000:  # لتفادي البطء
        sns.pairplot(df, hue=cat_cols[0] if len(cat_cols) > 0 else None, diag_kind="kde")
        plt.show()

    # ==============================
    # 7) Box & Violin & Swarm (Numeric vs Categorical)
    # ==============================
    for num_col in numeric_cols:
        for cat_col in cat_cols:
            plt.figure()
            sns.boxplot(x=cat_col, y=num_col, data=df, palette="Set2")
            plt.title(f"Boxplot of {num_col} by {cat_col}")
            plt.xticks(rotation=45)
            plt.show()

            plt.figure()
            sns.violinplot(x=cat_col, y=num_col, data=df, palette="muted")
            plt.title(f"Violin Plot of {num_col} by {cat_col}")
            plt.xticks(rotation=45)
            plt.show()

            plt.figure()
            sns.swarmplot(x=cat_col, y=num_col, data=df, palette="deep", size=4)
            plt.title(f"Swarm Plot of {num_col} by {cat_col}")
            plt.xticks(rotation=45)
            plt.show()

    # ==============================
    # 8) Scatter (Numeric vs Numeric)
    # ==============================
    if len(numeric_cols) > 1:
        for i in range(len(numeric_cols)):
            for j in range(i+1, len(numeric_cols)):
                plt.figure()
                sns.scatterplot(
                    x=df[numeric_cols[i]],
                    y=df[numeric_cols[j]],
                    hue=df[cat_cols[0]] if len(cat_cols) > 0 else None,
                    palette="tab10"
                )
                plt.title(f"Scatter: {numeric_cols[i]} vs {numeric_cols[j]}")
                plt.show()

    # ==============================
    # 9) Bar Plots (Mean of numeric by category)
    # ==============================
    for num_col in numeric_cols:
        for cat_col in cat_cols:
            plt.figure()
            sns.barplot(x=cat_col, y=num_col, data=df, ci="sd", palette="pastel")
            plt.title(f"Mean {num_col} by {cat_col}")
            plt.xticks(rotation=45)
            plt.show()

    # ==============================
    # 10) Line Plots (Datetime vs Numeric)
    # ==============================
    for col in datetime_cols:
        for num_col in numeric_cols:
            plt.figure()
            sns.lineplot(x=df[col], y=df[num_col], marker="o")
            plt.title(f"Line Plot of {num_col} over {col}")
            plt.show()

    print("\n✅ EDA Completed Successfully!")
