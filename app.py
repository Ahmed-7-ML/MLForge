# ---------------------------------
# Sidebar Navigation (Commented out, using tabs instead)
# ---------------------------------
# st.sidebar.title("App Journey")
# page = st.sidebar.radio("Go To", ["🏠 Home", "🧹 Data Cleaning", "📊 EDA", "🤖 Modeling", "🚀 Deployment"])

import streamlit as st
from pipeline.data import load_data, clean_data
from pipeline.eda import perform_eda
from pipeline.modeling import identify_problem, prepare_df, build_clustering_models
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
    
    # User selects the target column
    all_cols = st.session_state.df.columns.tolist()
    target = st.selectbox("Select target column", all_cols, key="target_select", help="The column to predict (not needed for Clustering).") if problem != "Clustering" else None
    
    # More user control: Select models, search type, number of trials
    available_models_dict = {
        "Classification": ['LogisticRegression', 'RandomForestClassifier', 'XGBClassifier', 'MLPClassifier', 'KNeighborsClassifier', 'SVC', 'DecisionTreeClassifier'],
        "Regression": ['LinearRegression', 'RandomForestRegressor', 'XGBRegressor', 'MLPRegressor', 'KNeighborsRegressor', 'SVR', 'DecisionTreeRegressor'],
        "Clustering": ['KMeans', 'DBSCAN']
    }

    available_models = available_models_dict.get(problem, [])
    if problem == "Clustering":
        # Changed to selectbox for single selection
        selected_models = st.selectbox("Select Algorithm", available_models, key="selected_models_key")
    else:
        selected_models = st.multiselect("Select Models to Train", available_models, default=available_models[:3], key="selected_models_key")

    search_type = st.selectbox("Hyperparameter Search Type", ["Random", "Grid", "Optuna"], key="search_type_key", help="Choose tuning method: Random (random search), Grid (exhaustive), Optuna (efficient Bayesian).")
    n_trials = st.slider("Number of Trials/Iterations", 5, 50, 10, key="n_trials_key", help="Number of hyperparameter combinations to try (ignored for Grid).")
    confirm_modeling = st.checkbox("Confirm: I understand training may take time.", help="Check to enable the start button for large actions.", key="confirm_modeling_key")

    if confirm_modeling and st.button("🚀 Start Modeling", type="primary", width='stretch'):
        if problem in ['Classification', 'Regression'] and not target:
            st.error("Please select a target column.")
        else:
            st.session_state.modeling_initiated = True
            with st.spinner("⏳ The models are being trained...it might take a minute or two"):
                progress_bar = st.progress(0)
                identify_problem(st.session_state.df, problem, target, selected_models, search_type, n_trials)
                progress_bar.progress(1.0)
            st.session_state.training_done = True

    if problem == 'Clustering':
        with st.spinner("Preparing data for clustering..."):
            X_train, X_test, _, _, _, _ = prepare_df(st.session_state.df)
            st.session_state.X_train_clust = X_train
            st.session_state.X_test_clust = X_test
            build_clustering_models(st.session_state.X_train_clust, st.session_state.X_test_clust, algorithm=selected_models)  # Now selected_models is str

    if st.session_state.get("best_models_trained") is not None or st.session_state.get("best_clustering_model") is not None:
        st.markdown("---")
        st.success("🎉 Training Ends Successfully, Now Save the Models")
        confirm_save = st.checkbox("Confirm: Save the best model for deployment.", key="confirm_save_key")
        if confirm_save and st.button("💾 Save Best Model for Deployment", type="primary", width='stretch'):
            target_encoder = st.session_state.target_encoder_saved
            feature_names_saved = st.session_state.feature_names_saved  # post-dummies
            original_feature_names = st.session_state.original_feature_names
            original_dtypes = st.session_state.original_dtypes
            problem_type = st.session_state.problem_type
            if problem_type == "clustering":
                best_model = st.session_state["best_clustering_model"]
                best_name = type(best_model).__name__
            else:
                best_models = st.session_state.best_models_trained
                if problem_type == "regression":
                    candidates = [
                        name for name in best_models
                        if "RandomForest" in name or "Linear" in name
                    ]
                    best_name = candidates[0] if candidates else list(best_models.keys())[0]
                else:  # classification
                    priority = ["RandomForestClassifier", "LogisticRegression",
                                "SVC", "KNeighborsClassifier", "DecisionTreeClassifier"]
                    best_name = next((name for name in priority if name in best_models), list(best_models.keys())[0])
                best_model = best_models.get(best_name)
            if best_model is None:
                st.error("Best model not found.")
            else:
                st.write(f"Selected Best Model: {best_name}")
                base_dir = os.path.dirname(os.path.abspath(__file__))
                models_dir = os.path.join(base_dir, "models")
                os.makedirs(models_dir, exist_ok=True)  # Ensure folder exists
                save_path = os.path.join(models_dir, "best_model.pkl")
                try:
                    with open(save_path, "wb") as f:
                        pickle.dump(best_model, f)
                    st.success(f"Model saved successfully at {save_path}")
                    st.write("File exists:", os.path.exists(save_path))
                except Exception as e:
                    st.error(f"Model saving failed: {e}")
                pickle.dump(feature_names_saved, open("models/feature_names.pkl", "wb"))  # post
                pickle.dump(original_feature_names, open("models/original_feature_names.pkl", "wb"))  # pre
                pickle.dump(original_dtypes, open("models/original_dtypes.pkl", "wb"))
                if target_encoder is not None:
                    pickle.dump(target_encoder, open("models/target_encoder.pkl", "wb"))
                for f in ["models/best_model.pkl", "models/feature_names.pkl", "models/original_feature_names.pkl", "models/original_dtypes.pkl", "models/target_encoder.pkl"]:
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
                if st.session_state.api_process is None:
                    st.error("API not running. Start the API first.")
                else:
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