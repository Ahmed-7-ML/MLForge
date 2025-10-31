
# ---------------------------------
# App Configuration
# ---------------------------------
# st.set_page_config(
#     page_title='ZEMASAi | Auto-ML',
#     page_icon='🤖',
#     layout='wide',
#     initial_sidebar_state='auto'
# )

# ---------------------------------
# Sidebar Naivgation
# ---------------------------------
# st.sidebar.title("App Journey")
# page = st.sidebar.radio("Go To", 
#     ["🏠 Home", "🧹 Data Cleaning", "📊 EDA", "🤖 Modeling", "🚀 Deployment"])

# ---------------------------------
# Main Page Content
# ---------------------------------
# st.title("ZEMASAi App")
# st.markdown("**Smoothly, Build your Model**")

# ---------------------------------
# Store uploaded data across pages
# ---------------------------------
# if "df" not in st.session_state:
#     st.session_state.df = None
#     st.session_state.cleaned = False


# ---------------------------------
# Home Page
# ---------------------------------
# if page == "🏠 Home":
#     st.write("🏡Welcome to ML Life Cycle Platform")
#     st.markdown("Upload your dataset to begin your machine learning journey.")
#     uploaded_file = st.sidebar.file_uploader("Select a File (CSV, Excel, JSON)", type=['csv', 'json', 'xls', 'xlsx'])

#     if uploaded_file is not None:
#         try:
#             df = load_data(uploaded_file)
#             # Save data in Session
#             st.session_state.df = df
#             st.session_state.cleaned = False
#             st.write("#### 🧾Raw Data Preview")
#             st.dataframe(df.head())
#         except ValueError as e:
#             st.error(str(e))

# ---------------------------------
# Data Cleaning Page
# ---------------------------------
# elif page == '🧹 Data Cleaning':
#     st.header("🧽Data Cleaning Stage")
#     if st.session_state.df is not None:
#         df_clean = clean_data(st.session_state.df)
#         # Update the Data
#         st.session_state.df = df_clean
#         st.session_state.cleaned = True
#         st.success("✅ Data cleaned successfully!")
#         st.write("### 🧾 Cleaned Data Preview")
#         st.dataframe(df_clean.head())
#     else:
#         st.warning("⚠️ Please upload data from the Home page first.")

# ---------------------------------
# Dashboard Page
# ---------------------------------
# elif page == "📊 EDA":
#     if st.session_state.df is not None:
#         perform_eda(st.session_state.df)
#     else:
#         st.warning("⚠️ Please upload and clean data before performing EDA.")

# ---------------------------------
# Modeling Page
# ---------------------------------
# elif page == "🤖 Modeling":
#     st.header("🤖 Build and Train Models")
#     if st.session_state.df is not None:
#         problem = st.selectbox(label='Enter your problem', options=['Classification', 'Regression', 'Clustering'])
#         # User selects the target column
#         all_columns = st.session_state.df.columns.tolist()
#         target = None
#         if problem in ["Classification", "Regression"]:
#             target = st.selectbox(label='Select the Target Column', options=all_columns)
        
#         if st.button("Start Modeling"):
#             if problem in ['Classification', 'Regression'] and not target:
#                 st.error("Please select a target column.")
#             else:
#                 try:
#                     identify_problem(st.session_state.df, target, problem)
#                 except Exception as e:
#                     st.error(f"Modeling error: {e}")
#     else:
#         st.warning("⚠️ Please upload and clean data before modeling.")

# ---------------------------------
# Deployment Page
# ---------------------------------
# elif page == "🚀 Deployment":
#     st.header("🚀 Deployment and Prediction")
#     if st.session_state.df is not None:
#         st.info("Deploy your trained model here.")
#     else:
#         st.warning("⚠️ Please complete the previous steps first.")

import streamlit as st
import pandas as pd
from pipeline.data import load_data, clean_data
from pipeline.eda import perform_eda, handle_chat_query
from pipeline.modeling import identify_problem

st.set_page_config(
    page_title='ZEMASAi | Auto-ML',
    page_icon='🤖',
    layout='wide',
)

# ---------------------------------
# Session State Init
# ---------------------------------
if "df" not in st.session_state:
    st.session_state.df = None
    st.session_state.cleaned = False
if "chat_open" not in st.session_state:
    st.session_state.chat_open = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------------------------------
# Tabs
# ---------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["🏠 Home", "🧹 Data Cleaning", "📊 EDA", "🤖 Modeling", "🚀 Deployment"]
)

with tab1:
    st.title("🏡 Welcome to ML Life Cycle Platform")
    uploaded_file = st.file_uploader(
        "Upload dataset", type=['csv', 'json', 'xlsx'])
    if uploaded_file:
        df = load_data(uploaded_file)
        st.session_state.df = df
        st.session_state.cleaned = False
        st.dataframe(df.head())

with tab2:
    st.header("🧽 Data Cleaning")
    if st.session_state.df is not None:
        df_clean = clean_data(st.session_state.df)
        st.session_state.df = df_clean
        st.session_state.cleaned = True
        st.dataframe(df_clean.head())
    else:
        st.warning("Upload data first")

with tab3:
    if st.session_state.df is not None:
        perform_eda(st.session_state.df)
    else:
        st.warning("Upload data first")

with tab4:
    st.header("🤖 Modeling")
    if st.session_state.df is not None:
        problem = st.selectbox(
            "Problem Type", ["Classification", "Regression", "Clustering"])
        all_cols = st.session_state.df.columns.tolist()
        target = st.selectbox(
            "Select target", all_cols) if problem != "Clustering" else None

        if st.button("Start Modeling"):
            identify_problem(st.session_state.df, target, problem)
    else:
        st.warning("Upload data first")

with tab5:
    st.header("🚀 Deployment Coming Soon")