import matplotlib.pyplot as plt
import seaborn as sns


def perform_eda(df):
    """
    Perform Exploratory Data Analysis (EDA) on the dataset.
    Includes summary info and multiple visualizations.
    """

    # --- Basic Info ---
    print("\n--- Dataset Info ---")
    print(df.info())
    print("\n--- Missing Values ---")
    print(df.isnull().sum())
    print("\n--- Descriptive Statistics ---")
    print(df.describe(include='all').T)

    # --- Visualizations ---
    numeric_cols = df.select_dtypes(include="number").columns
    categorical_cols = df.select_dtypes(include="object").columns

    # 1. Distribution plots for numeric features
    for col in numeric_cols:
        plt.figure(figsize=(6, 4))
        sns.histplot(df[col], kde=True, bins=20, color="steelblue")
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.tight_layout()
        plt.show()

    # 2. Count plots for categorical features
    for col in categorical_cols:
        plt.figure(figsize=(6, 4))
        sns.countplot(x=col, data=df, palette="viridis")
        plt.title(f"Count of {col}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    # 3. Correlation heatmap
    if len(numeric_cols) > 1:
        plt.figure(figsize=(8, 6))
        corr = df[numeric_cols].corr()
        sns.heatmap(corr, annot=True, cmap="coolwarm", linewidths=0.5)
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.show()

    # 4. Pairplot (scatter matrix with hue)
    if len(numeric_cols) > 1 and len(df) < 2000:
        sns.pairplot(df, hue=categorical_cols[0] if len(categorical_cols) > 0 else None, diag_kind="kde")
        plt.show()

    # 5. Boxplots (numeric vs categorical)
    for num_col in numeric_cols:
        for cat_col in categorical_cols:
            plt.figure(figsize=(6, 4))
            sns.boxplot(x=cat_col, y=num_col, data=df, palette="Set2")
            plt.title(f"Boxplot of {num_col} by {cat_col}")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

    # 6. Violin plots (distribution by category)
    for num_col in numeric_cols:
        for cat_col in categorical_cols:
            plt.figure(figsize=(6, 4))
            sns.violinplot(x=cat_col, y=num_col, data=df, palette="muted")
            plt.title(f"Violin Plot of {num_col} by {cat_col}")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

    # 7. Scatter plots (numeric vs numeric)
    if len(numeric_cols) > 1:
        for i in range(len(numeric_cols)):
            for j in range(i+1, len(numeric_cols)):
                plt.figure(figsize=(6, 4))
                sns.scatterplot(
                    x=df[numeric_cols[i]],
                    y=df[numeric_cols[j]],
                    hue=df[categorical_cols[0]] if len(categorical_cols) > 0 else None,
                    palette="tab10"
                )
                plt.title(f"Scatter: {numeric_cols[i]} vs {numeric_cols[j]}")
                plt.tight_layout()
                plt.show()

    # 8. Bar plots (mean of numeric by category)
    for num_col in numeric_cols:
        for cat_col in categorical_cols:
            plt.figure(figsize=(6, 4))
            sns.barplot(x=cat_col, y=num_col, data=df, ci="sd", palette="pastel")
            plt.title(f"Mean {num_col} by {cat_col}")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

    # 9. Swarm plots (scatter + categories)
    for num_col in numeric_cols:
        for cat_col in categorical_cols:
            plt.figure(figsize=(6, 4))
            sns.swarmplot(x=cat_col, y=num_col, data=df, palette="deep", size=4)
            plt.title(f"Swarm Plot of {num_col} by {cat_col}")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

    # 10. Line plots
    datetime_cols = df.select_dtypes(include="datetime").columns
    for col in datetime_cols:
        for num_col in numeric_cols:
            plt.figure(figsize=(8, 4))
            sns.lineplot(x=df[col], y=df[num_col], marker="o")
            plt.title(f"Line Plot of {num_col} over {col}")
            plt.tight_layout()
            plt.show()
