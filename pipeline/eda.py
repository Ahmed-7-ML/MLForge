import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


def perform_eda(df, selected_num_cols=None, selected_cat_cols=None):
    """
    Perform Exploratory Data Analysis (EDA) on a DataFrame.
    """
    # Use selected columns, otherwise default to all
    numeric_cols = selected_num_cols if selected_num_cols else df.select_dtypes(
        include=[np.number]).columns.tolist()
    cat_cols = selected_cat_cols if selected_cat_cols else df.select_dtypes(
        include=["object", "category"]).columns.tolist()

    st.write(
        f"Analyzing {len(numeric_cols)} numeric and {len(cat_cols)} categorical columns.")

    # --- Numeric Distributions (Histograms and Boxplots) ---
    if numeric_cols:
        st.subheader("🔢 Numeric Distributions")
        nbins = st.radio("Select Number of Bins for Histograms", options=[5, 10, 15, 20], index=1)

        # Determine number of rows for subplots
        num_rows_hist = len(numeric_cols)

        fig = make_subplots(
            rows=num_rows_hist, cols=2,
            subplot_titles=[f"Histogram of {col}" for col in numeric_cols] +
            [f"Boxplot of {col}" for col in numeric_cols],
            column_widths=[0.6, 0.4]
        )

        for i, col in enumerate(numeric_cols):
            # Histogram
            hist = px.histogram(df, x=col, nbins=nbins)
            for trace in hist['data']:
                fig.add_trace(trace, row=i+1, col=1)

            # Boxplot
            box = px.box(df, y=col)
            for trace in box['data']:
                fig.add_trace(trace, row=i+1, col=2)

        fig.update_layout(height=400 * num_rows_hist,
                          title_text="Numeric Distributions (Histogram & Boxplot)")
        st.plotly_chart(fig, use_container_width=True)

    # --- Categorical Distributions ---
    if cat_cols:
        st.markdown("---")
        st.subheader("📝 Categorical Distributions")

        num_cols_per_row = 3  # To keep the plots clean
        num_rows_cat = (len(cat_cols) + num_cols_per_row -
                        1) // num_cols_per_row

        fig_cat = make_subplots(
            rows=num_rows_cat,
            cols=num_cols_per_row,
            subplot_titles=[f"Countplot of {col}" for col in cat_cols]
        )

        for i, col in enumerate(cat_cols):
            row = (i // num_cols_per_row) + 1
            col_pos = (i % num_cols_per_row) + 1

            bar = px.bar(df, x=col)
            for trace in bar['data']:
                fig_cat.add_trace(trace, row=row, col=col_pos)

        fig_cat.update_layout(height=400 * num_rows_cat,
                              title_text="Categorical Distributions")
        st.plotly_chart(fig_cat, use_container_width=True)

    # --- Correlation Heatmap (Matplotlib) ---
    if len(numeric_cols) > 1:
        st.markdown("---")
        st.subheader("📊 Correlation Heatmap")
        corr = df[numeric_cols].corr()
        plt.figure(figsize=(7, 5))
        sns.heatmap(corr, annot=True, fmt=".2f",
                    cmap="coolwarm", linewidths=0.5)
        plt.title("Correlation Heatmap")
        st.pyplot(plt.gcf())
        plt.clf()

    # --- Scatter Plots (Numeric vs Numeric) ---
    if len(numeric_cols) > 1:
        st.markdown("---")
        st.subheader(
            "📈 Scatter Plots (Relationships between Numeric Variables)")

        # Dynamic creation of scatter plots
        selected_pairs = st.multiselect(
            "Select pairs for scatter plots",
            options=[(col1, col2) for i, col1 in enumerate(numeric_cols)
                     for col2 in numeric_cols[i+1:]]
        )

        if selected_pairs:
            fig_scatter = make_subplots(rows=len(selected_pairs), cols=1,
                                        subplot_titles=[f"{pair[0]} vs {pair[1]}" for pair in selected_pairs])

            for i, (x_col, y_col) in enumerate(selected_pairs):
                scatter = px.scatter(df, x=x_col, y=y_col,
                                     color=cat_cols[0] if cat_cols else None)
                for trace in scatter['data']:
                    fig_scatter.add_trace(trace, row=i+1, col=1)

            fig_scatter.update_layout(
                height=400 * len(selected_pairs), title_text="Scatter Plots")
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.info("Select at least one pair of columns to generate scatter plots.")

    st.success("✅ EDA Completed Successfully!")
