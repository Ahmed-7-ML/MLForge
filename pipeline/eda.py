# ---------------------------------
# Exploratory Data Analysis Script
# Render AI Charts
# Chatbot Query Handler
# Auto Charts (Fallback)
# Main perform_eda (Custom EDA, AI Dashboard,Manual Charts and Chatbot)
# ---------------------------------
import numpy as np
import plotly.express as px
import streamlit as st
import re
import json
import os
from dotenv import load_dotenv

# Gemini Setup
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except Exception:
    GEMINI_AVAILABLE = False

# Load .env (only in local, ignored on Streamlit Cloud)
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)

# Get API key: Prioritize Streamlit secrets for cloud deployment, fallback to env for local
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY") if 'secrets' in dir(st) else os.getenv('GEMINI_API_KEY')

# Default to fast model
GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'gemini-1.5-flash')
if GEMINI_AVAILABLE and GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
    except Exception as e:
        st.error(f"Gemini configuration error: {e}")
        GEMINI_AVAILABLE = False
else:
    GEMINI_AVAILABLE = False

# ----------------------------------------------------
# Chatbot Query Handler
# ----------------------------------------------------
def handle_chat_query(query, df):
    if not GEMINI_AVAILABLE:
        return "Gemini is not available. Please set GEMINI_API_KEY in Streamlit secrets or .env."

    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        summary = df.describe(include='all').to_json()
        cols = list(df.columns)
        prompt = f"""
        You are an expert ML engineer and data scientist.
        Dataset has {df.shape[0]} rows and columns: {cols}
        Answer this question precisely and professionally:
        Question: {query}
        Summary statistics:
        {summary}
        If asked about modeling:
        - Suggest problem type (Classification/Regression/Clustering)
        - Recommend best models with reasons
        - Suggest feature engineering tips
        - Assist in Hyperparameter Tuning
        """
        response = model.generate_content(prompt)
        text = response.text if hasattr(response, 'text') else str(response.candidates[0].content.parts[0].text)
        return text.strip()
    except Exception as e:
        return f"Gemini error: {str(e)}"

# ----------------------------------------------------
# Auto Charts (Fallback)
# ----------------------------------------------------
def auto_charts(df, numeric_cols, cat_cols):
    st.write("### Automatically Generated Charts")

    date_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.datetime64)]
    if len(date_cols) > 0 and len(numeric_cols) > 0:
        st.plotly_chart(px.line(df, x=date_cols[0], y=numeric_cols[0], title="Trend Over Time"), use_container_width=True)

    for c in cat_cols[:3]:
        if len(numeric_cols) > 0:
            grouped = df.groupby(c, as_index=False)[numeric_cols[0]].mean()
            st.plotly_chart(px.bar(grouped, x=c, y=numeric_cols[0], title=f"Avg {numeric_cols[0]} by {c}"), use_container_width=True)

    if len(cat_cols) > 0 and len(numeric_cols) > 0:
        g = df.groupby(cat_cols[0], as_index=False)[numeric_cols[0]].sum()
        st.plotly_chart(px.pie(g, names=cat_cols[0], values=numeric_cols[0], title=f"{numeric_cols[0]} Distribution"), use_container_width=True)

    if len(numeric_cols) > 0:
        st.plotly_chart(px.histogram(df, x=numeric_cols[0], title=f"Histogram - {numeric_cols[0]}"), use_container_width=True)

    if len(cat_cols) > 0 and len(numeric_cols) > 0:
        st.plotly_chart(px.box(df, x=cat_cols[0], y=numeric_cols[0], title=f"{numeric_cols[0]} by {cat_cols[0]}"), use_container_width=True)

    if len(numeric_cols) >= 2:
        st.plotly_chart(px.scatter(df, x=numeric_cols[0], y=numeric_cols[1], title=f"{numeric_cols[0]} vs {numeric_cols[1]}"), use_container_width=True)

    if len(numeric_cols) > 1:
        corr = df[numeric_cols].corr()
        fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", aspect="auto")
        fig.update_layout(title="Correlation Heatmap")
        st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------
# Render AI Charts
# ----------------------------------------------------
def render_charts_from_plan(df, charts):
    for i, ch in enumerate(charts):
        try:
            fig_type = ch.get("type", "scatter").lower()
            x = ch.get("x")
            y = ch.get("y")
            color = ch.get("color")
            title = ch.get("title", f"Chart {i+1}")

            if not x or (fig_type != "pie" and not y):
                st.warning(f"Skipping invalid chart: {ch}")
                continue

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
                fig = px.scatter(df, x=x, y=y, color=color, title=title)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Failed to render chart: {e}")

# ----------------------------------------------------
# Main perform_eda
# ----------------------------------------------------
def perform_eda(df):
    if df is None or df.empty:
        st.warning("No data available. Please upload and clean your dataset first.")
        return

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    tab1, tab2, tab3, tab4 = st.tabs(["Data Profile", "AI Dashboard", "Manual Charts", "Chatbot"])

    # ==================== TAB 1: Custom EDA ====================
    with tab1:
        st.header("Exploratory Data Analysis")
        inner_tabs = st.tabs(["Overview", "Numeric", "Categorical", "Correlations", "Missing"])

        with inner_tabs[0]: # Overview
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Rows", f"{df.shape[0]:,}")
            col2.metric("Columns", f"{df.shape[1]:,}")
            col3.metric("Missing", f"{df.isnull().sum().sum():,}")
            col4.metric("Duplicates", f"{df.duplicated().sum():,}")
            st.write("### Preview")
            st.dataframe(df.head(10), use_container_width=True)
            st.write("### Data Types")
            st.dataframe(df.dtypes.reset_index().rename(
                columns={0: "dtype"}), use_container_width=True)

        with inner_tabs[1]: # Numerical Columns
            if len(numeric_cols) == 0:
                st.info("No numeric columns.")
            else:
                col = st.selectbox(
                    "Select numeric", numeric_cols, key="num_sel")
                c1, c2 = st.columns(2)
                with c1:
                    st.plotly_chart(px.histogram(
                        df, x=col, title=f"Distribution - {col}"), use_container_width=True)
                with c2:
                    st.plotly_chart(
                        px.box(df, y=col, title=f"Boxplot - {col}"), use_container_width=True)
                st.dataframe(df[col].describe(), use_container_width=True)

        with inner_tabs[2]: # Categorical Columns
            if len(cat_cols) == 0:
                st.info("No categorical columns.")
            else:
                col = st.selectbox("Select categorical", cat_cols, key="cat_sel")
                c1, c2 = st.columns(2)
                with c1:
                    counts = df[col].value_counts().head(10)
                    st.plotly_chart(px.bar(x=counts.index, y=counts.values, title=f"Top 10 - {col}"), use_container_width=True)
                with c2:
                    st.plotly_chart(px.pie(values=counts.values, names=counts.index, title=col), use_container_width=True)

        with inner_tabs[3]: # Correlations
            if len(numeric_cols) < 2:
                st.info("Need 2+ numeric columns.")
            else:
                corr = df[numeric_cols].corr()
                st.plotly_chart(px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", title="Correlation"), use_container_width=True)

        with inner_tabs[4]: # Missings
            missing = df.isnull().sum()
            missing = missing[missing > 0]
            if missing.empty:
                st.success("No missing values!")
            else:
                st.plotly_chart(px.bar(x=missing.index, y=missing.values, title="Missing Values"), use_container_width=True)

    # ==================== TAB 2: AI Dashboard ====================
    with tab2:
        st.write("### Generate AI-Powered Dashboard")
        if st.button("Generate AI Dashboard", type="primary"):
            with st.spinner("Gemini is thinking..."):
                if not GEMINI_AVAILABLE:
                    st.warning("Gemini unavailable → Using auto charts")
                    st.session_state["ai_charts"] = "auto"
                else:
                    try:
                        prompt = f"""
                        You are an expert data visualization analyst.
                        Return ONLY a valid JSON array (no extra text, no markdown, no explanation) containing exactly 6 insightful Plotly charts.
                        Use this exact format (no backticks, no json word):

                        [
                        {{"type": "scatter", "x": "age", "y": "income",
                            "color": "gender", "title": "Income vs Age by Gender"}},
                        {{"type": "histogram", "x": "age", "color": "churn",
                            "title": "Age Distribution by Churn"}},
                        ...
                        ]

                        Data columns: {list(df.columns)}
                        Numeric columns: {numeric_cols}
                        Categorical columns: {cat_cols}

                        Return only the JSON array. Nothing else.
                        """
                        model = genai.GenerativeModel("gemini-2.5-flash")
                        resp = model.generate_content(prompt)
                        json_text = re.search(r'\[[\s\S]*\]', resp.text, re.DOTALL)
                        if not json_text:
                            raise ValueError("No JSON found")
                        charts = json.loads(json_text.group(0))
                        st.session_state["ai_charts"] = charts
                        st.success("AI Dashboard ready!")
                    except Exception as e:
                        st.error(f"AI failed: {e}")
                        st.session_state["ai_charts"] = "auto"

        if st.session_state.get("ai_charts"):
            if st.session_state["ai_charts"] == "auto":
                auto_charts(df, numeric_cols, cat_cols)
            else:
                render_charts_from_plan(df, st.session_state["ai_charts"])

    # ==================== TAB 3: Manual Charts ====================
    with tab3:
        st.write("### Create Custom Chart")
        chart_type = st.selectbox("Type", ["Scatter", "Line", "Bar", "Pie", "Box"])
        x = st.selectbox("X", df.columns)
        y = st.selectbox("Y", [None] + numeric_cols) if chart_type != "Pie" else None
        color = st.selectbox("Color by", ["None"] + list(df.columns))
        if st.button("Plot"):
            color_val = None if color == "None" else color
            if chart_type == "Scatter":
                fig = px.scatter(df, x=x, y=y, color=color_val)
            elif chart_type == "Line":
                fig = px.line(df, x=x, y=y, color=color_val)
            elif chart_type == "Bar":
                fig = px.bar(df, x=x, y=y, color=color_val)
            elif chart_type == "Pie":
                fig = px.pie(df, names=x, values=y)
            elif chart_type == "Box":
                fig = px.box(df, x=x, y=y, color=color_val)
            st.plotly_chart(fig, use_container_width=True)

    # ==================== TAB 4: Chatbot ====================
    with tab4:
        st.write("### Chat with Your Data Assistant and ML Advisor")
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        for msg in st.session_state.chat_history:
            st.write(f"**You:** {msg['q']}")
            st.write(f"**Gemini:** {msg['a']}")

        query = st.text_area("Ask anything about your data or ML advice:", height=100)
        if st.button("Send"):
            if query.strip():
                reply = handle_chat_query(query, df)
                st.session_state.chat_history.append({"q": query, "a": reply})
                st.rerun()
