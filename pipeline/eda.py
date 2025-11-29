# ---------------------------------
# Enhanced EDA Dashboard with Tabs
# 1 — Data Profile
# 2 — AI Dashboard
# 3 — Manual Visualization
# 4 — Chatbot Query
# ---------------------------------
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import streamlit as st
from ydata_profiling import ProfileReport
import streamlit.components.v1 as components
import re
import json
import os
from dotenv import load_dotenv
from pipeline.data import clean_data

# Gemini Setup -------------------------
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except Exception:
    GEMINI_AVAILABLE = False

dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path)
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
GEMINI_MODEL = os.getenv('GEMINI_MODEL')

if GEMINI_AVAILABLE and GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    GEMINI_AVAILABLE = False


# ----------------------------------------------------
# Chatbot Query Handler (Fallback for complex queries)
# ----------------------------------------------------
def handle_chat_query(query, df):
    q = query.strip().lower()
    cols = list(df.columns)
    # Column detection
    found = None
    for c in cols:
        if c.lower() in q:
            found = c
            break

    # Local answers
    if found:
        if "mean" in q or "average" in q:
            if pd.api.types.is_numeric_dtype(df[found]):
                return f"Mean of '{found}' = {df[found].mean():.4f}"

        if "null" in q or "missing" in q:
            return f"Missing values in '{found}': {df[found].isnull().sum()}"

        if "unique" in q:
            return f"Unique values in '{found}': {df[found].nunique()}"

    # Gemini fallback for any query
    if GEMINI_AVAILABLE:
        try:
            model = genai.GenerativeModel(GEMINI_MODEL)
            resp = model.generate_content(
                f"Analyze this user question about the dataset: {query}\n"
                f"Available columns: {cols}\n"
                f"Provide a precise answer based on data stats. \
                If it involves computation like correlation between specific columns, compute it conceptually."
            )
            return getattr(resp, "text", None) or resp.candidates[0].content.parts[0].text
        except Exception as e:
            return f"Gemini Error: {str(e)}"

    return "I couldn't understand. Try asking: mean age / unique gender / correlation between age and salary."

# ----------------------------------------------------
# ----------------------------------------------------
def auto_charts(df, numeric_cols, cat_cols):
    st.write("### 📈 Automatically Generated Charts (Enhanced Fallback)")

    # Datetime line chart
    date_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.datetime64)]
    if date_cols and numeric_cols:
        st.plotly_chart(
            px.line(df, x=date_cols[0], y=numeric_cols[0],
                    title="Line Chart Over Time"),
            use_container_width=True
        )

    # Bar charts for first 3 cat cols
    for c in cat_cols[:3]:
        if numeric_cols:
            grouped = df.groupby(c, as_index=False)[numeric_cols[0]].mean()
            st.plotly_chart(
                px.bar(
                    grouped, x=c, y=numeric_cols[0], title=f"Average {numeric_cols[0]} by {c}"),
                use_container_width=True
            )

    # Pie chart
    if cat_cols and numeric_cols:
        g = df.groupby(cat_cols[0], as_index=False)[numeric_cols[0]].sum()
        st.plotly_chart(
            px.pie(g, names=cat_cols[0], values=numeric_cols[0],
                title=f"Distribution of {numeric_cols[0]} by {cat_cols[0]}"),
            use_container_width=True)

    # Histogram for first numeric col
    if numeric_cols:
        st.plotly_chart(
            px.histogram(
                df, x=numeric_cols[0], title=f"Histogram of {numeric_cols[0]}"),
            use_container_width=True
        )

    # Boxplot for first cat and num
    if cat_cols and numeric_cols:
        st.plotly_chart(
            px.box(df, x=cat_cols[0], y=numeric_cols[0],
                title=f"Boxplot of {numeric_cols[0]} by {cat_cols[0]}"),
            use_container_width=True
        )

    # Scatter for first two num
    if len(numeric_cols) >= 2:
        st.plotly_chart(
            px.scatter(df, x=numeric_cols[0], y=numeric_cols[1],
                    title=f"Scatter of {numeric_cols[0]} vs {numeric_cols[1]}"),
            use_container_width=True
        )

    # Correlation matrix
    if len(numeric_cols) > 1:
        corr = df[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        cax = ax.matshow(corr, vmin=-1, vmax=1, cmap='coolwarm')
        plt.xticks(range(len(numeric_cols)), numeric_cols, rotation=90)
        plt.yticks(range(len(numeric_cols)), numeric_cols)
        plt.colorbar(cax)
        st.pyplot(fig)

# ----------------------------------------------------
# Render AI Suggested Charts
# ----------------------------------------------------
def render_charts_from_plan(df, charts):
    for ch in charts:
        try:
            fig_type = ch.get("type", "")
            x, y = ch.get("x"), ch.get("y")
            color = ch.get("color")
            title = ch.get("title", "Chart")

            if fig_type == "bar":
                fig = px.bar(df, x=x, y=y, color=color, title=title)
            elif fig_type == "line":
                fig = px.line(df, x=x, y=y, color=color, title=title)
            elif fig_type == "scatter":
                fig = px.scatter(df, x=x, y=y, color=color, title=title)
            elif fig_type == "pie":
                fig = px.pie(df, names=x, values=y, title=title)
            elif fig_type == "box":
                fig = px.box(df, x=x, y=y, color=color, title=title)
            else:
                continue

            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error rendering chart: {str(e)}")

def smart_data_chatbot(query, df):
    q = query.lower().strip()
    cols = df.columns

    # Enhanced parsing for multiple columns (e.g., correlation between col1 and col2)
    # Use regex to extract potential columns
    potential_cols = [c for c in cols if c.lower() in q]

    # ======================
    # NUMERIC QUESTIONS
    # ======================
    if any(k in q for k in ["mean", "average"]) and potential_cols:
        target_col = potential_cols[0]
        if pd.api.types.is_numeric_dtype(df[target_col]):
            return f"Mean of '{target_col}' = {df[target_col].mean():.3f}"

    if "median" in q and potential_cols:
        target_col = potential_cols[0]
        if pd.api.types.is_numeric_dtype(df[target_col]):
            return f"Median of '{target_col}' = {df[target_col].median():.3f}"

    if "std" in q or "standard deviation" in q and potential_cols:
        target_col = potential_cols[0]
        if pd.api.types.is_numeric_dtype(df[target_col]):
            return f"STD of '{target_col}' = {df[target_col].std():.3f}"

    if "max" in q and potential_cols:
        target_col = potential_cols[0]
        return f"Maximum of '{target_col}' is {df[target_col].max()}"

    if "min" in q and potential_cols:
        target_col = potential_cols[0]
        return f"Minimum of '{target_col}' is {df[target_col].min()}"

    # ======================
    # GENERAL COLUMN QUESTIONS
    # ======================
    if "null" in q or "missing" in q and potential_cols:
        target_col = potential_cols[0]
        return f"Missing values in '{target_col}': {df[target_col].isnull().sum()}"

    if "unique" in q and potential_cols:
        target_col = potential_cols[0]
        uniq = df[target_col].nunique()
        return f"Unique values in '{target_col}': {uniq}"

    if "describe" in q and potential_cols:
        target_col = potential_cols[0]
        return str(df[target_col].describe())

    if "columns" in q:
        return f"Available columns:\n{list(cols)}"

    if "shape" in q:
        return f"Dataset shape: {df.shape}"

    if "head" in q:
        return df.head().to_string()

    if "correlation" in q:
        if len(potential_cols) >= 2:
            col1, col2 = potential_cols[:2]
            if pd.api.types.is_numeric_dtype(df[col1]) and pd.api.types.is_numeric_dtype(df[col2]):
                corr_value = df[col1].corr(df[col2])
                return f"Correlation between '{col1}' and '{col2}': {corr_value:.3f}"
        else:
            nums = df.select_dtypes(include='number').columns
            if len(nums) >= 2:
                return df[nums].corr().to_string()
            return "No numeric columns to compute correlation."

    # ======================
    # RELATION QUESTIONS
    # ======================
    if "relationship" in q or "correlate" in q:
        if len(potential_cols) >= 2:
            col1, col2 = potential_cols[:2]
            if pd.api.types.is_numeric_dtype(df[col1]) and pd.api.types.is_numeric_dtype(df[col2]):
                corr_value = df[col1].corr(df[col2])
                return f"Correlation between '{col1}' and '{col2}': {corr_value:.3f}"
        else:
            nums = df.select_dtypes(include='number').columns
            if len(nums) >= 2:
                return df[nums].corr().to_string()
            return "No numeric columns to compute correlation."

    # ======================
    # FALLBACK TO GEMINI FOR COMPLEX QUERIES
    # ======================
    return handle_chat_query(query, df)  # Use Gemini for anything not caught

# ----------------------------------------------------
# Main perform_eda — Now Completely Tab-based
# ----------------------------------------------------
def perform_eda(df):
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Data Profile",
        "🤖 AI Dashboard",
        "🎨 Manual Charts",
        "💬 Chatbot"
    ])

    # ===============================================
    # TAB 1 — Data Profile
    # ===============================================
    with tab1:
        st.write("### 📊 Full Pandas Profiling Report")

        if st.button("Generate Data Profile"):
            profile = ProfileReport(df, title="EDA Report", explorative=True)
            st.session_state["profile_html"] = profile.to_html()
            # Export to file for download
            profile.to_file("eda_report.html")

        if "profile_html" in st.session_state:
            components.html(st.session_state["profile_html"], height=1000, scrolling=True)
            # Download buttons
            with open("eda_report.html", "rb") as f:
                st.download_button("Download HTML Report", f, file_name="eda_report.html")

    # ===============================================
    # TAB 2 — AI Dashboard
    # ===============================================
    with tab2:
        st.write("### 🤖 Generate AI Suggested Dashboard")

        if st.button("✨ Generate AI Dashboard"):
            if not GEMINI_AVAILABLE:
                st.warning("Gemini not available → Showing Enhanced Auto Charts")
                st.session_state["ai_charts"] = "auto"
            else:
                try:
                    summary = df.describe(include='all').transpose().to_json()
                    prompt = f"""
                    You are a data scientist assistant.
                    Understand the data and Suggest 8 Plotly chart plans based on this data summary.
                    Make the graphic charts understandable and expressive of the data.
                    Return JSON list with: type, x, y, color, title.
                    Summary: {summary}
                    """

                    model = genai.GenerativeModel(GEMINI_MODEL)
                    resp = model.generate_content(prompt)
                    raw = getattr(resp, "text", None) or resp.candidates[0].content.parts[0].text
                    match = re.search(r"\[.*\]", raw, re.DOTALL)
                    charts = json.loads(match.group(0)) if match else []
                    if not charts:
                        raise ValueError("No charts generated")
                    st.session_state["ai_charts"] = charts

                except Exception as e:
                    st.error(f"Gemini failed: {str(e)}. Falling back to enhanced auto charts.")
                    st.session_state["ai_charts"] = "auto"

        if "ai_charts" in st.session_state:
            if st.session_state["ai_charts"] == "auto":
                auto_charts(df, numeric_cols, cat_cols)
            else:
                render_charts_from_plan(df, st.session_state["ai_charts"])

    # ===============================================
    # TAB 3 — Manual Visualization
    # ===============================================
    with tab3:
        st.write("### 🎨 Create Manual Plotly Charts")

        chart_type = st.selectbox("Select Chart Type", ["Scatter", "Line", "Bar", "Pie", "Box"])

        x = st.selectbox("X-axis", df.columns)
        y = None

        if chart_type not in ["Pie"]:
            y = st.selectbox("Y-axis", numeric_cols)

        color = st.selectbox("Color (optional)", ["None"] + list(df.columns))

        if st.button("Generate Chart"):
            if chart_type == "Scatter":
                fig = px.scatter(df, x=x, y=y, color=None if color == "None" else color)
            elif chart_type == "Line":
                fig = px.line(df, x=x, y=y, color=None if color =="None" else color)
            elif chart_type == "Bar":
                fig = px.bar(df, x=x, y=y, color=None if color =="None" else color)
            elif chart_type == "Pie":
                fig = px.pie(df, names=x, values=y if y else None)
            elif chart_type == "Box":
                fig = px.box(df, x=x, y=y, color=None if color =="None" else color)

            st.plotly_chart(fig, use_container_width=True)

    # ===============================================
    # TAB 4 — Chatbot
    # ===============================================
    with tab4:
        st.write("### 💬 Chat with Your Data")

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Show previous messages
        for h in st.session_state.chat_history:
            st.markdown(f"**You:** {h['q']}")
            st.markdown(f"**Assistant:** {h['a']}")

        user_msg = st.text_input("Ask a question about your data:")

        if st.button("Ask"):
            bot_reply = smart_data_chatbot(user_msg, df)
            st.session_state.chat_history.append({"q": user_msg, "a": bot_reply})
            st.rerun()
