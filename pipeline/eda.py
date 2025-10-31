# # ====================================
# # Exploratory Data Analysis Script
# # ====================================
# # This dashboard provides:
# #   1. Numerical Variables Histograms and Box Plots
# #   2. Categorical Variables Bar plot vs Target and Count Plot
# #   3. Bivariate Analysis -> Scatter Plot between Variables
# #   4. Multivariate Analysis -> Correlation Matrix
# #   5. AI-Powered Dashboard (Gemini + Streamlit)
# # ====================================
# import pandas as pd
# import numpy as np
# import plotly.express as px
# import matplotlib.pyplot as plt
# import streamlit as st
# import re
# import json
# import os
# from dotenv import load_dotenv

# # try import Gemini; if not present we'll fallback to local answers
# try:
#     import google.generativeai as genai
#     GEMINI_AVAILABLE = True
# except Exception:
#     GEMINI_AVAILABLE = False

# # load env
# dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
# load_dotenv(dotenv_path)
# GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
# GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'models/gemini-1.5-flash')

# if GEMINI_AVAILABLE and GEMINI_API_KEY:
#     genai.configure(api_key=GEMINI_API_KEY)
# else:
#     # ensure we know we don't have a working Gemini
#     GEMINI_AVAILABLE = False


# def perform_eda(df):
#     """AI-powered EDA (Gemini when available). Chatbot falls back to local compute for simple stats."""
#     st.title("🤖 AI-Powered Exploratory Data Analysis Dashboard")
#     st.write(f"**Rows:** {df.shape[0]} | **Columns:** {df.shape[1]}")
#     st.dataframe(df.head())

#     numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
#     cat_cols = df.select_dtypes(
#         include=['object', 'category']).columns.tolist()

#     # quick stats panel
#     with st.expander("Quick Statistics"):
#         st.write(df.describe(include='all').transpose())
#         st.write("Missing per column:")
#         st.write(df.isnull().sum())

#     # AI Dashboard
#     if st.button("✨ Generate AI Dashboard"):
#         if not GEMINI_AVAILABLE:
#             st.warning(
#                 "Gemini not available — showing simple automatic charts instead.")
#             auto_charts(df, numeric_cols, cat_cols)
#         else:
#             with st.spinner("Calling Gemini to suggest visualizations..."):
#                 try:
#                     summary = df.describe(include='all').transpose().to_json()
#                     prompt = f"""You are a data viz assistant. Given the dataset summary below (JSON), suggest 3-5 Plotly charts as a JSON list with fields: type,x,y,color,title. Summary: {summary}"""
#                     model = genai.GenerativeModel(GEMINI_MODEL)
#                     resp = model.generate_content(prompt)
#                     raw = getattr(
#                         resp, "text", None) or resp.candidates[0].content.parts[0].text
#                     match = re.search(r"\[.*\]", raw, re.DOTALL)
#                     charts = json.loads(match.group(0)) if match else []
#                     if not charts:
#                         st.info(
#                             "Gemini returned no valid chart plan — showing auto charts.")
#                         auto_charts(df, numeric_cols, cat_cols)
#                     else:
#                         render_charts_from_plan(df, charts)
#                 except Exception as e:
#                     # handle common Gemini errors gracefully
#                     msg = str(e)
#                     if "429" in msg:
#                         st.warning(
#                             "Gemini quota exceeded — showing automatic charts instead.")
#                     elif "404" in msg or "not found" in msg.lower():
#                         st.warning(
#                             "Gemini model not found — showing automatic charts instead.")
#                     else:
#                         st.warning(
#                             f"Gemini error: {e} — falling back to automatic charts.")
#                     auto_charts(df, numeric_cols, cat_cols)

#     st.markdown("---")
#     # Chatbot UI (fallback supported)
#     st.markdown(
#         "<style>.chat-btn{position:fixed;bottom:18px;right:18px;z-index:9999;}</style>", unsafe_allow_html=True)
#     if "chat_open" not in st.session_state:
#         st.session_state.chat_open = False
#     if st.button("💬 Ask ZEMASAi", key="open_chat"):
#         st.session_state.chat_open = not st.session_state.chat_open

#     if st.session_state.chat_open:
#         with st.container():
#             st.markdown("### 🤖 ZEMASAi Chat")
#             if "chat_history" not in st.session_state:
#                 st.session_state.chat_history = []
#             q = st.text_input(
#                 "Ask about your data (e.g. 'price average', 'count nulls in age')", key="chat_input")
#             if st.button("Send", key="chat_send"):
#                 if not q or not q.strip():
#                     st.warning("Enter a question.")
#                 else:
#                     answer = handle_chat_query(q, df)
#                     st.session_state.chat_history.append((q, answer))
#             for user, bot in reversed(st.session_state.chat_history[-8:]):
#                 st.markdown(f"**You:** {user}")
#                 st.markdown(f"**ZEMASAi:** {bot}")


# def auto_charts(df, numeric_cols, cat_cols):
#     """Simple auto charts when Gemini not available."""
#     st.subheader("Automatic charts")
#     # show up to 4 numeric histograms
#     for c in numeric_cols[:4]:
#         st.plotly_chart(px.histogram(df, x=c, nbins=30,
#                         title=f"Histogram of {c}"), use_container_width=True)
#     # show up to 4 categorical counts
#     for c in cat_cols[:4]:
#         counts = df[c].value_counts().reset_index()
#         counts.columns = [c, 'count']
#         st.plotly_chart(px.bar(counts, x=c, y='count',
#                         title=f"Counts of {c}"), use_container_width=True)
#     # correlation if enough numeric
#     if len(numeric_cols) > 1:
#         corr = df[numeric_cols].corr()
#         fig, ax = plt.subplots(figsize=(7, 5))
#         cax = ax.matshow(corr, vmin=-1, vmax=1)
#         plt.xticks(range(len(numeric_cols)), numeric_cols, rotation=90)
#         plt.yticks(range(len(numeric_cols)), numeric_cols)
#         plt.colorbar(cax)
#         st.pyplot(fig)


# def render_charts_from_plan(df, charts):
#     st.subheader("📊 AI Suggestions")
#     for ch in charts:
#         ctype = ch.get("type", "bar")
#         x = ch.get("x")
#         y = ch.get("y")
#         color = ch.get("color")
#         title = ch.get("title", "Chart")
#         if x not in df.columns:
#             continue
#         try:
#             if ctype == "bar":
#                 fig = px.bar(df, x=x, y=y, color=color, title=title)
#             elif ctype == "line":
#                 fig = px.line(df, x=x, y=y, color=color, title=title)
#             elif ctype == "pie":
#                 fig = px.pie(df, names=x, values=y, title=title)
#             elif ctype == "scatter":
#                 fig = px.scatter(df, x=x, y=y, color=color, title=title)
#             elif ctype == "box":
#                 fig = px.box(df, x=x, y=y, color=color, title=title)
#             elif ctype == "histogram":
#                 fig = px.histogram(df, x=x, color=color, title=title)
#             else:
#                 continue
#             st.plotly_chart(fig, use_container_width=True)
#         except Exception:
#             continue


# def handle_chat_query(query, df):
#     """Try Gemini if available; else attempt local answer for simple questions."""
#     # simple intent parsing: look for 'average' / 'mean' / 'count nulls' / 'unique' etc.
#     q = query.strip().lower()
#     # try local quick answers first for simple ops
#     # find column name in query by checking df columns
#     cols = list(df.columns)
#     found = None
#     for c in cols:
#         if c.lower() in q:
#             found = c
#             break

#     # average/mean
#     if any(k in q for k in ["average", "avg", "mean"]):
#         if found and pd.api.types.is_numeric_dtype(df[found]):
#             val = df[found].mean()
#             return f"Average (mean) of '{found}' = {round(float(val), 4)}"
#         else:
#             # fallback to Gemini if available
#             pass

#     # count nulls
#     if any(k in q for k in ["null", "missing", "nan"]):
#         if found:
#             cnt = int(df[found].isnull().sum())
#             return f"Missing values in '{found}': {cnt}"
#     # unique values
#     if "unique" in q or "distinct" in q:
#         if found:
#             vals = df[found].nunique(dropna=True)
#             return f"'{found}' has {vals} unique values."

#     # otherwise try Gemini if available
#     if GEMINI_AVAILABLE:
#         try:
#             model = genai.GenerativeModel(GEMINI_MODEL)
#             resp = model.generate_content(
#                 f"Answer briefly: {query}\nColumns: {cols}")
#             text = getattr(
#                 resp, "text", None) or resp.candidates[0].content.parts[0].text
#             return text.strip()
#         except Exception as e:
#             msg = str(e)
#             if "429" in msg:
#                 return "⚠️ Gemini quota exceeded — please wait a bit, or ask a simple stats question (e.g. 'price average')."
#             if "404" in msg or "not found" in msg.lower():
#                 return "⚠️ Gemini model not found — check GEMINI_MODEL in your .env."
#             return f"⚠️ Gemini error: {e}"
#     else:
#         return "⚠️ Gemini not configured — try a simple question like 'price average' so I can compute locally."
# ====================================
# Exploratory Data Analysis Script
# ====================================
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import streamlit as st
import re
import json
import os
from dotenv import load_dotenv

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except Exception:
    GEMINI_AVAILABLE = False

dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path)
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'models/gemini-1.5-flash')

if GEMINI_AVAILABLE and GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    GEMINI_AVAILABLE = False


def perform_eda(df):
    st.title("🤖 AI-Powered Exploratory Data Analysis Dashboard")
    st.write(f"**Rows:** {df.shape[0]} | **Columns:** {df.shape[1]}")
    st.dataframe(df.head())

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(
        include=['object', 'category']).columns.tolist()

    with st.expander("Quick Statistics"):
        st.write(df.describe(include='all').transpose())
        st.write("Missing per column:")
        st.write(df.isnull().sum())

    if st.button("✨ Generate AI Dashboard"):
        if not GEMINI_AVAILABLE:
            st.warning(
                "Gemini not available — showing simple automatic charts.")
            auto_charts(df, numeric_cols, cat_cols)
        else:
            try:
                summary = df.describe(include='all').transpose().to_json()
                prompt = f"""You are a data viz assistant. 
                Suggest 3-5 Plotly charts as JSON.
                Summary: {summary}"""

                model = genai.GenerativeModel(GEMINI_MODEL)
                resp = model.generate_content(prompt)
                raw = getattr(
                    resp, "text", None) or resp.candidates[0].content.parts[0].text

                match = re.search(r"\[.*\]", raw, re.DOTALL)
                charts = json.loads(match.group(0)) if match else []
                if not charts:
                    auto_charts(df, numeric_cols, cat_cols)
                else:
                    render_charts_from_plan(df, charts)

            except Exception as e:
                st.warning(f"Gemini error: {e}")
                auto_charts(df, numeric_cols, cat_cols)


def auto_charts(df, numeric_cols, cat_cols):
    st.subheader("Automatic charts")

    for c in numeric_cols[:4]:
        st.plotly_chart(px.histogram(
            df, x=c, title=f"Histogram of {c}"), use_container_width=True)

    for c in cat_cols[:4]:
        counts = df[c].value_counts().reset_index()
        counts.columns = [c, 'count']
        st.plotly_chart(px.bar(counts, x=c, y='count',
                        title=f"Counts of {c}"), use_container_width=True)

    if len(numeric_cols) > 1:
        corr = df[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(7, 5))
        cax = ax.matshow(corr, vmin=-1, vmax=1)
        plt.xticks(range(len(numeric_cols)), numeric_cols, rotation=90)
        plt.yticks(range(len(numeric_cols)), numeric_cols)
        plt.colorbar(cax)
        st.pyplot(fig)


def render_charts_from_plan(df, charts):
    st.subheader("📊 AI Suggestions")
    for ch in charts:
        try:
            fig = getattr(px, ch.get("type", "bar"))(
                df,
                x=ch.get("x"),
                y=ch.get("y"),
                color=ch.get("color"),
                title=ch.get("title", "Chart")
            )
            st.plotly_chart(fig, use_container_width=True)
        except:
            continue


def handle_chat_query(query, df):
    q = query.strip().lower()
    cols = list(df.columns)
    found = None

    for c in cols:
        if c.lower() in q:
            found = c
            break

    if any(k in q for k in ["average", "mean"]):
        if found and pd.api.types.is_numeric_dtype(df[found]):
            return f"Mean of '{found}' = {df[found].mean():.4f}"

    if "null" in q or "missing" in q:
        if found:
            return f"Missing values in '{found}': {df[found].isnull().sum()}"

    if "unique" in q:
        if found:
            return f"Unique values in '{found}': {df[found].nunique()}"

    if GEMINI_AVAILABLE:
        try:
            model = genai.GenerativeModel(GEMINI_MODEL)
            resp = model.generate_content(f"Answer: {query}\nColumns: {cols}")
            return getattr(resp, "text", None) or resp.candidates[0].content.parts[0].text
        except:
            return "Gemini Error."

    return "Local mode: ask stats like 'age mean'"
