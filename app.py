import streamlit as st
import pandas as pd
from pipeline.data import load_data, clean_data
from pipeline.eda import perform_eda

# ---------------------------------
# App Configuration
# ---------------------------------
st.set_page_config(
    page_title='ZEMASAi | Auto-ML',
    page_icon='🤖',
    layout='wide',
    initial_sidebar_state='auto'
)

# ---------------------------------
# Sidebar Naivgation
# ---------------------------------
st.sidebar.title("App Journey")
page = st.sidebar.radio("Go To", 
    ["🏠 Home", "🧹 Data Cleaning", "📊 EDA", "🤖 Modeling", "🚀 Deployment"])

# ---------------------------------
# Main Page Content
# ---------------------------------
st.title("ZEMASAi App")
st.markdown("**Smoothly, Build your Model**")

# ---------------------------------
# Store uploaded data across pages
# ---------------------------------
if "df" not in st.session_state:
    st.session_state.df = None


# ---------------------------------
# Home Page
# ---------------------------------
if page == "🏠 Home":
    st.write("🏡Welcome to ML Life Cycle Platform")
    st.markdown("Upload your dataset to begin your machine learning journey.")
    uploaded_file = st.sidebar.file_uploader("Select a File (CSV, Excel, JSON)", type=['csv', 'json', 'xls', 'xlsx'])

    if uploaded_file is not None:
        try:
            df = load_data(uploaded_file)
            # Save data in Session
            st.session_state.df = df
            st.write("#### 🧾Raw Data Preview")
            st.dataframe(df.head())
        except ValueError as e:
            st.error(str(e))

# ---------------------------------
# Data Cleaning Page
# ---------------------------------
elif page == '🧹 Data Cleaning':
    st.header("🧽Data Cleaning Stage")
    if st.session_state.df is not None:
        df_clean = clean_data(st.session_state.df)
        # Update the Data
        st.session_state.df = df_clean
        st.success("✅ Data cleaned successfully!")
        st.write("### 🧾 Cleaned Data Preview")
        st.dataframe(df_clean.head())
    else:
        st.warning("⚠️ Please upload data from the Home page first.")

# ---------------------------------
# Dashboard Page
# ---------------------------------
elif page == "📊 EDA":
    st.header("📈 Exploratory Data Analysis (EDA)")
    if st.session_state.df is not None:
        perform_eda(st.session_state.df)
    else:
        st.warning("⚠️ Please upload and clean data before performing EDA.")

# ---------------------------------
# Modeling Page
# ---------------------------------
elif page == "🤖 Modeling":
    st.header("🤖 Build and Train Models")
    if st.session_state.df is not None:
        st.info("Model training and evaluation will appear here soon.")
    else:
        st.warning("⚠️ Please upload and clean data before modeling.")

# ---------------------------------
# Deployment Page
# ---------------------------------
elif page == "🚀 Deployment":
    st.header("🚀 Deployment and Prediction")
    if st.session_state.df is not None:
        st.info("Deploy your trained model here.")
    else:
        st.warning("⚠️ Please complete the previous steps first.")
