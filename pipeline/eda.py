# ---------------------------------
# Exploratory Data Analysis Script:-
# Enhanced EDA Dashboard with Tabs
# Modifications: Enhanced chatbot to use Gemini for all queries, support multiple questions, provide model building advice based on data (e.g., suggest problem type, best models).
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

# Gemini Setup
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
# Chatbot Query Handler (Now fully Gemini-based, supports model advice)
# ----------------------------------------------------
def handle_chat_query(query, df):
    if not GEMINI_AVAILABLE:
        return "Gemini not available. Please check API key."
    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        summary = df.describe(include='all').transpose().to_json()
        cols = list(df.columns)
        prompt = f"""
        You are an expert data scientist and ML advisor.
        Analyze this user question about the dataset: {query}
        Available columns: {cols}
        Dataset summary: {summary}
        Provide a precise answer. For data stats (mean, correlation, etc.), compute conceptually.
        If the question is about building models:
        - Suggest problem type (Regression/Classification/Clustering) based on data.
        - Recommend best models and why (e.g., based on data size, imbalance).
        - Advise on hyperparameters, features, etc.
        Support multiple questions in one session.
        """
        resp = model.generate_content(prompt)
        return getattr(resp, "text", None) or resp.candidates[0].content.parts[0].text
    except Exception as e:
        return f"Gemini Error: {str(e)}"

# ----------------------------------------------------
# Auto Charts (fallback) - Enhanced with more charts
# ----------------------------------------------------
def auto_charts(df, numeric_cols, cat_cols):
    st.write("### 📈 Automatically Generated Charts")
    # Datetime line chart
    date_cols = [c for c in df.columns if np.issubdtype(
        df[c].dtype, np.datetime64)]
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
# Render AI Suggested Charts (Gemini)
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

# ----------------------------------------------------
# Main perform_eda — Now Completely Tab-based
# ----------------------------------------------------
def perform_eda(df):
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(
        include=['object', 'category']).columns.tolist()
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
    # with tab1:
    #     st.write("### 📊 Full Pandas Profiling Report")
    #     if st.button("Generate Data Profile"):
    #         profile = ProfileReport(df, title="EDA Report", explorative=True)
    #         st.session_state["profile_html"] = profile.to_html()
    #         # Export to file for download
    #         profile.to_file("eda_report.html")
    #     if "profile_html" in st.session_state:
    #         components.html(st.session_state["profile_html"], height=1000, scrolling=True)
    #         # Download buttons
    #         with open("eda_report.html", "rb") as f:
    #             st.download_button("Download HTML Report", f, file_name="eda_report.html")

    with tab1:
        st.header("Exploratory Data Analysis")
        if st.session_state.df is None:
            st.warning("Please upload data first.")
            st.stop()

        df = st.session_state.df

        tabs = st.tabs(["Overview", "Numeric Features",
                    "Categorical Features", "Correlations", "Missing & Duplicates"])

        with tabs[0]:  # Overview
            st.subheader("Dataset Overview")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Rows", f"{df.shape[0]:,}")
            with col2:
                st.metric("Columns", f"{df.shape[1]:,}")
            with col3:
                st.metric("Missing Values", f"{df.isnull().sum().sum():,}")
            with col4:
                st.metric("Duplicates", f"{df.duplicated().sum():,}")

            st.write("### First 10 Rows")
            st.dataframe(df.head(10), use_container_width=True)

            st.write("### Data Types")
            st.dataframe(pd.DataFrame(df.dtypes).T, use_container_width=True)

        with tabs[1]:  # Numeric Features
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                st.info("No numeric columns found.")
            else:
                st.subheader("Numeric Features Distribution")
                selected_num = st.selectbox(
                    "Select column", numeric_cols, key="num_col")

                col1, col2 = st.columns(2)
                with col1:
                    fig = px.histogram(df, x=selected_num, nbins=30,
                                    color_discrete_sequence=['#636EFA'])
                    fig.update_layout(title=f"Distribution of {selected_num}")
                    st.plotly_chart(fig, use_container_width=True)
                with col2:
                    fig = px.box(df, y=selected_num,
                                color_discrete_sequence=['#EF553B'])
                    fig.update_layout(title=f"Box Plot - {selected_num}")
                    st.plotly_chart(fig, use_container_width=True)

                st.write("### Statistical Summary")
                st.dataframe(df[selected_num].describe(), use_container_width=True)

        with tabs[2]:  # Categorical Features
            cat_cols = df.select_dtypes(include=['object', 'category']).columns
            if len(cat_cols) == 0:
                st.info("No categorical columns found.")
            else:
                st.subheader("Categorical Features")
                selected_cat = st.selectbox(
                    "Select column", cat_cols, key="cat_col")

                col1, col2 = st.columns(2)
                with col1:
                    counts = df[selected_cat].value_counts().head(10)
                    fig = px.bar(x=counts.index, y=counts.values, labels={
                                'x': selected_cat, 'y': 'Count'})
                    fig.update_layout(title=f"Top 10 Values - {selected_cat}")
                    st.plotly_chart(fig, use_container_width=True)
                with col2:
                    fig = px.pie(values=counts.values, names=counts.index,
                                title=f"Distribution - {selected_cat}")
                    st.plotly_chart(fig, use_container_width=True)

        with tabs[3]:  # Correlations
            num_df = df.select_dtypes(include=[np.number])
            if num_df.shape[1] < 2:
                st.info("Not enough numeric columns for correlation.")
            else:
                st.subheader("Correlation Matrix")
                corr = num_df.corr()
                fig = px.imshow(corr, text_auto=True, aspect="auto",color_continuous_scale="RdBu_r")
                fig.update_layout(title="Correlation Heatmap")
                st.plotly_chart(fig, use_container_width=True)

                # Top correlated pairs
                corr_pairs = corr.unstack().sort_values(ascending=False)
                corr_pairs = corr_pairs[corr_pairs != 1.0].drop_duplicates()
                st.write("### Strongest Correlations")
                st.dataframe(corr_pairs.head(10).to_frame(
                    name="Correlation"), use_container_width=True)

        with tabs[4]:  # Missing & Duplicates
        st.subheader("Missing Values")
        missing = df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        if missing.empty:
            st.success("No missing values!")
        else:
            fig = px.bar(x=missing.index, y=missing.values, labels={'x': 'Column', 'y': 'Missing Count'})
            fig.update_layout(title="Missing Values per Column")
            st.plotly_chart(fig, use_container_width=True)

        if df.duplicated().sum() > 0:
            st.error(f"{df.duplicated().sum():,} duplicate rows found!")
            if st.button("Show Duplicates"):
                st.dataframe(df[df.duplicated(keep=False)], use_container_width=True)

    # ===============================================
    # TAB 2 — AI Dashboard
    # ===============================================
    with tab2:
        st.write("### 🤖 Generate AI Suggested Dashboard")
        if st.button("✨ Generate AI Dashboard"):
            if not GEMINI_AVAILABLE:
                st.warning(
                    "Gemini not available → Showing Enhanced Auto Charts")
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
                    raw = getattr(
                        resp, "text", None) or resp.candidates[0].content.parts[0].text
                    match = re.search(r"\[.*\]", raw, re.DOTALL)
                    charts = json.loads(match.group(0)) if match else []
                    if not charts:
                        raise ValueError("No charts generated")
                    st.session_state["ai_charts"] = charts
                except Exception as e:
                    st.error(
                        f"Gemini failed: {str(e)}. Falling back to enhanced auto charts.")
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
                fig = px.scatter(
                    df, x=x, y=y, color=None if color == "None" else color)
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
    # TAB 4 — Chatbot (Enhanced for multi-questions and model advice)
    # ===============================================
    with tab4:
        st.write("### 💬 Chat with Your Data & ML Advisor")
        st.markdown(
            "Ask about data stats, correlations, or ML advice (e.g., 'What model is best for this data?')")
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        # Show previous messages
        for h in st.session_state.chat_history:
            st.markdown(f"**You:** {h['q']}")
            st.markdown(f"**Bot:** {h['a']}")
        user_msg = st.text_area("Ask questions (multiple lines OK):")
        if st.button("Send"):
            if user_msg:
                # Split into multiple questions if separated by newlines or semicolons
                questions = re.split(r'\n|;', user_msg)
                for q in questions:
                    q = q.strip()
                    if q:
                        bot_reply = handle_chat_query(q, df)
                        st.session_state.chat_history.append(
                            {"q": q, "a": bot_reply})
                st.rerun()
