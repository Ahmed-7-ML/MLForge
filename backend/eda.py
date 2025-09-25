import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st 

sns.set_style(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)


def perform_eda(df, selected_num_cols=None, selected_cat_cols=None):
    """
    Perform Exploratory Data Analysis (EDA) on a DataFrame.
    """

    # ==============================
    # 1) Basic Info
    # ==============================
    st.write("### 📊 Dataset Overview")
    st.write("**Shape:**", df.shape)
    st.write("**Data Types:**")
    st.write(df.dtypes)
    st.write("**Missing Values:**")
    st.write(df.isnull().sum())
    st.write("**Summary Statistics:**")
    st.write(df.describe(include="all").T)

    # ==============================
    # 2) Split Columns
    # ==============================
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(
        include=["object", "category"]).columns.tolist()
    datetime_cols = df.select_dtypes(include="datetime").columns.tolist()

    if not selected_num_cols:
        selected_num_cols = numeric_cols
    if not selected_cat_cols:
        selected_cat_cols = cat_cols

    # ==============================
    # 3) Numeric Features
    # ==============================
    for col in selected_num_cols:
        plt.figure()
        sns.histplot(df[col].dropna(), kde=True, bins=30, color="steelblue")
        plt.title(f"Distribution of {col}")
        st.pyplot(plt.gcf())
        plt.clf()

        plt.figure()
        sns.boxplot(x=df[col], color="skyblue")
        plt.title(f"Boxplot of {col}")
        st.pyplot(plt.gcf())
        plt.clf()

    # ==============================
    # 4) Categorical Features
    # ==============================
    for col in selected_cat_cols:
        plt.figure()
        sns.countplot(
            x=df[col], order=df[col].value_counts().index, palette="viridis")
        plt.title(f"Countplot of {col}")
        plt.xticks(rotation=45)
        st.pyplot(plt.gcf())
        plt.clf()

    # ==============================
    # 5) Correlation Heatmap
    # ==============================
    if len(selected_num_cols) > 1:
        plt.figure(figsize=(12, 8))
        corr = df[selected_num_cols].corr()
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Heatmap")
        st.pyplot(plt.gcf())
        plt.clf()

    # ==============================
    # 6) Pairplot
    # ==============================
    if 1 < len(selected_num_cols) <= 6 and len(df) < 2000:
        pairplot = sns.pairplot(df[selected_num_cols + selected_cat_cols],
                                hue=selected_cat_cols[0] if selected_cat_cols else None,
                                diag_kind="kde")
        st.pyplot(pairplot.fig)
        plt.clf()

    # ==============================
    # 7) Box & Violin (Numeric vs Categorical)
    # ==============================
    for num_col in selected_num_cols:
        for cat_col in selected_cat_cols:
            plt.figure()
            sns.boxplot(x=cat_col, y=num_col, data=df, palette="Set2")
            plt.title(f"Boxplot of {num_col} by {cat_col}")
            plt.xticks(rotation=45)
            st.pyplot(plt.gcf())
            plt.clf()

            plt.figure()
            sns.violinplot(x=cat_col, y=num_col, data=df, palette="muted")
            plt.title(f"Violin Plot of {num_col} by {cat_col}")
            plt.xticks(rotation=45)
            st.pyplot(plt.gcf())
            plt.clf()

    # ==============================
    # 8) Scatter (Numeric vs Numeric)
    # ==============================
    if len(selected_num_cols) > 1:
        for i in range(len(selected_num_cols)):
            for j in range(i+1, len(selected_num_cols)):
                plt.figure()
                sns.scatterplot(
                    x=df[selected_num_cols[i]],
                    y=df[selected_num_cols[j]],
                    hue=df[selected_cat_cols[0]] if selected_cat_cols else None,
                    palette="tab10"
                )
                plt.title(
                    f"Scatter: {selected_num_cols[i]} vs {selected_num_cols[j]}")
                st.pyplot(plt.gcf())
                plt.clf()

    st.success("✅ EDA Completed Successfully!")
