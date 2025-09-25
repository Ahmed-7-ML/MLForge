import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from backend.data import load_data, clean_data
from backend.eda import perform_eda

# Set plot style
sns.set_style(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# App configuration
st.set_page_config(page_title="Interactive EDA App", layout="wide")
st.title("📊 Interactive Data Uploader & EDA Dashboard")

# Session state management
if 'df' not in st.session_state:
    st.session_state.df = None
if 'df_cleaned' not in st.session_state:
    st.session_state.df_cleaned = None

# --- Main App Logic ---
# Sidebar for file upload
with st.sidebar:
    st.header("⚙️ Settings")
    file = st.file_uploader("Upload your Tabular file", type=[
                            "csv", "xlsx", "xls", "json"])

    if st.button("Reset App"):
        st.session_state.df = None
        st.session_state.df_cleaned = None
        st.rerun()

# Load file into session state and perform automatic cleaning
if file:
    file_name = file.name.lower()
    if st.session_state.df is None or st.session_state.df.empty or st.session_state.get('file_name') != file_name:
        try:
            st.session_state.df = load_data(file)

            with st.spinner("Performing automatic data cleaning..."):
                st.session_state.df_cleaned = clean_data(st.session_state.df)
            st.session_state['file_name'] = file_name
            st.success("File uploaded and data cleaned successfully!")
        except Exception as e:
            st.error(f"Error reading or cleaning file: {e}")
            st.session_state.df = None
else:
    st.info("Awaiting file upload...")

# Main content
df = st.session_state.df_cleaned

if df is not None and not df.empty:
    tab_overview, tab_eda = st.tabs(
        ["📝 Data Overview & Cleaning", "📈 EDA Visualizer"])

    with tab_overview:
        st.header("Data Overview & Cleaning")

        st.subheader("Dataset Information")
        st.write("Shape:", df.shape)
        st.write("Data Types:")
        st.dataframe(df.dtypes, use_container_width=True)

        st.subheader("Summary Statistics")
        st.dataframe(df.describe(include="all").T, use_container_width=True)

        st.subheader("Sample of Cleaned Data")
        n_rows = st.slider(
            "Select number of rows to display", 5, 50, 10, step=5)
        st.dataframe(df.head(n_rows), use_container_width=True)

    with tab_eda:
        st.header("Exploratory Data Analysis")
        perform_eda(df)
