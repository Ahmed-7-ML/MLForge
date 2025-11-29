# ---------------------------------
# Sidebar Navigation (Commented out, using tabs instead)
# ---------------------------------
# st.sidebar.title("App Journey")
# page = st.sidebar.radio("Go To", ["🏠 Home", "🧹 Data Cleaning", "📊 EDA", "🤖 Modeling", "🚀 Deployment"])

import streamlit as st
import pandas as pd
from pipeline.data import load_data, clean_data
from pipeline.eda import perform_eda
from pipeline.modeling import identify_problem
import subprocess
import time
import os
import io
import pickle
import sys
import requests  # Added for API calls in inference UI

# ---------------------------------
# App Configuration
# ---------------------------------
st.set_page_config(
    page_title='ML Forge | Auto-ML',
    page_icon='🤖',
    layout='wide',
    initial_sidebar_state='auto'
)

# ---------------------------------
# Main Page Content
# ---------------------------------
st.title("ML Forge App")
st.markdown("**Smoothly, Build your Model**")
st.info("This platform automates the ML lifecycle: upload data, clean, explore, model, and deploy. Follow the tabs step-by-step.")

# ---------------------------------
# Session State Init -> Store uploaded data across pages
# ---------------------------------
if "df" not in st.session_state:
    st.session_state.df = None
    st.session_state.cleaned = False
    st.session_state.cleaned_df = None
if "chat_open" not in st.session_state:
    st.session_state.chat_open = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "api_process" not in st.session_state:
    st.session_state.api_process = None  # To store the API subprocess

# ---------------------------------
# Tabs
# ---------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["🏠 Home", "🧹 Data Cleaning", "📊 EDA", "🤖 Modeling", "🚀 Deployment"]
)

# ---------------------------------
# Home Page (Data Upload)
# ---------------------------------
with tab1:
    st.header("🏡 Welcome to ML Life Cycle Platform")
    st.markdown("Upload your dataset here to start the pipeline.")
    uploaded_file = st.file_uploader("Upload dataset", type=['csv', 'json', 'xlsx'], help="Supported formats: CSV, JSON, Excel. Data will be loaded and previewed.")
    if uploaded_file is not None:
        try:
            df = load_data(uploaded_file)
            # Save data in Session
            st.session_state.df = df
            st.session_state.cleaned = False
            st.write("#### 🧾 Raw Data Preview")
            st.dataframe(df.head())
            st.write(f"Shape = {df.shape}")
            st.write("#### Descriptive Statistics")
            st.dataframe(df.describe(include='all'))
            st.write("#### MetaData")
            buffer = io.StringIO()
            df.info(buf=buffer)
            st.text(buffer.getvalue())
            st.write("##### Missing Values")
            st.table(df.isna().sum())
            st.write(f"##### Number of Duplicates Values = {df.duplicated().sum()}")
        except ValueError as e:
            st.error(str(e))

# ---------------------------------
# Data Cleaning Page
# ---------------------------------
with tab2:
    st.header("🧽 Data Cleaning")
    if st.session_state.df is not None:
        df_clean, clean_log, run_id = clean_data(st.session_state.df)  # returns log and run_id
        # Update the Data
        st.session_state.cleaned_df = df_clean
        st.session_state.cleaned = True
        st.success("✨ Data Cleaning Completed Successfully!")
        st.write("### 📝 Cleaning Summary")
        st.json(clean_log)
        st.download_button("Download Cleaned Data", df_clean.to_csv(index=False).encode('utf-8'), "cleaned_data.csv")
        st.info(f"MLflow Run ID: {run_id} - Check MLflow UI for logged parameters.")
        st.write("### 🧾 Cleaned Data Preview")
        st.dataframe(df_clean.head())
        st.write(f"Shape = {df.shape}")
    else:
        st.warning("⚠️ Please upload data from the Home page first.")

# ---------------------------------
# Dashboard Page
# ---------------------------------
with tab3:
    st.header("📊 AI-Powered Exploratory Data Analysis Dashboard")
    st.markdown("Explore your data with statistics, visualizations, and AI suggestions.")
    if st.session_state.df is not None:
        perform_eda(st.session_state.cleaned_df)
    else:
        st.warning("⚠️ Please upload and clean data before performing EDA.")

# ---------------------------------
# Modeling Page
# ---------------------------------
with tab4:
    st.header("🤖 Build and Train Models")
    st.markdown("Select problem type and target, then train multiple models automatically.")

    if st.session_state.df is None:
        st.warning("⚠️ Please upload and clean data before modeling.")
        st.stop()

    problem = st.selectbox("Problem Type", ["Classification", "Regression", "Clustering"], key="problem_select", help="Choose the ML task: Classification (categories), Regression (numbers), Clustering (groups).")

    all_cols = st.session_state.df.columns.tolist()     # User selects the target column
    target = st.selectbox("Select target column", all_cols, key="target_select",help="The column to predict (not needed for Clustering).") if problem != "Clustering" else None

    confirm_modeling = st.checkbox("Confirm: I understand training may take time.",help="Check to enable the start button for large actions.")
    if confirm_modeling and st.button("🚀 Start Modeling", type="primary", use_container_width=True):
        if problem in ['Classification', 'Regression'] and not target:
            st.error("Please select a target column.")
        elif problem == 'Clustering' and not target:
            progress_bar = st.progress(0)  # Added progress bar
            identify_problem(st.session_state.df, problem)
            progress_bar.progress(1.0)  # Complete after training
        else:
            with st.spinner("⏳ The models are being trained...it might take a minute or two"):
                progress_bar = st.progress(0)  # Added progress bar
                # Simulate progress (since training is in another function, update after call)
                identify_problem(st.session_state.df, problem, target)
                progress_bar.progress(1.0)  # Complete after training

    if st.session_state.get("best_models_trained") is not None:
        st.markdown("---")
        st.success("🎉 Training Ends Successfully, Now Save the Models")

        confirm_save = st.checkbox("Confirm: Save the best model for deployment.")
        if confirm_save and st.button("💾 Save Best Model for Deployment", type="primary", use_container_width=True):
            os.makedirs("models", exist_ok=True)
            best_models = st.session_state.best_models_trained
            target_encoder = st.session_state.target_encoder_saved
            feature_names = st.session_state.feature_names_saved
            problem_type = st.session_state.problem_type

            if problem_type == "regression":
                candidates = [name for name in best_models if "RandomForest" in name or "Linear" in name]
                best_name = candidates[0] if candidates else list(best_models.keys())[0]
            else:  # classification
                priority = ["RandomForestClassifier", "LogisticRegression","SVC", "KNeighborsClassifier", "DecisionTreeClassifier"]
                best_name = next((name for name in priority if name in best_models), list(best_models.keys())[0])

            best_model = best_models[best_name]
            pickle.dump(best_model, open("models/best_model.pkl", "wb"))
            pickle.dump(list(feature_names), open("models/feature_names.pkl", "wb"))
            if target_encoder is not None:
                pickle.dump(target_encoder, open("models/target_encoder.pkl", "wb"))

            for f in ["models/best_model.pkl", "models/feature_names.pkl", "models/target_encoder.pkl"]:
                if os.path.exists(f):
                    st.success(f"✅ {f} saved successfully!")
            st.success(f"🚀**{best_name}**: Model Saved Successfully")
            st.balloons()
            st.info("Go to the **Deployment** tab and click 'Run API' now!")

# ---------------------------------
# Deployment Page
# ---------------------------------
with tab5:
    st.subheader("🚀 Deploy Trained Model as REST API")
    st.markdown("Deploy your model as an API and test predictions directly here.")
    if not os.path.exists("models/best_model.pkl"):
        st.warning("⚠️ No trained model found. Please complete Modeling step first.")
    else:
        if st.button("🚀 Run API", help="Starts the FastAPI server locally."):
            st.info("Starting FastAPI server on http://127.0.0.1:8000")
            process = subprocess.Popen([sys.executable, "-m", "uvicorn", "pipeline.api:app", "--host", "127.0.0.1", "--port", "8000", "--reload"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
                )
            st.session_state.api_process = process  # Store process in session
            time.sleep(3)
            st.success("🎉 API is LIVE!")
            st.balloons()
            st.markdown("### 📖 Open API Documentation:")
            st.markdown("[Swagger UI - http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)")
            st.code("http://127.0.0.1:8000/docs", language="text")

        # Stop button for API (Added as per requirements)
        if st.session_state.api_process is not None and st.button("🛑 Stop API", help="Stops the running FastAPI server."):
            st.session_state.api_process.kill()
            st.session_state.api_process = None
            st.success("API stopped successfully!")

        # Inference UI: Form for input features and predict via API (Added as per requirements)
        st.subheader("🔍 Test Model Inference")
        st.markdown("Enter feature values below and get predictions from the deployed API.")
        if os.path.exists("models/feature_names.pkl"):
            feature_names = pickle.load(open("models/feature_names.pkl", "rb"))
            inputs = {}
            for feature in feature_names:
                inputs[feature] = st.number_input(f"{feature}", help=f"Enter value for {feature}.", value=0.0)

            if st.button("Predict", help="Send inputs to the API for prediction."):
                try:
                    payload = {"data": [inputs]}
                    response = requests.post("http://127.0.0.1:8000/predict", json=payload)
                    if response.status_code == 200:
                        result = response.json()
                        st.success(f"Prediction: {result['predictions']}")
                    else:
                        st.error(f"Error: {response.text}")
                except requests.exceptions.ConnectionError:
                    st.error("API not running. Start the API first.")
        else:
            st.warning("Feature names not found. Save a model first.")

    # Add Docker button
    # if st.button("Build Docker Image", help="Builds a Docker image for the app (requires Docker installed)."):
    #     try:
    #         subprocess.run(
    #             ["docker", "build", "-t", "ml-forge", "."], check=True)
    #         st.success(
    #             "Docker image built successfully! Run with: docker run -p 8501:8501 ml-forge")
    #     except Exception as e:
    #         st.error(f"Docker build error: {e}")
