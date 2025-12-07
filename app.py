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
    st.header("Build and Train Models")
    st.markdown("Select problem type and target, then train multiple models automatically.")

    if st.session_state.df is None:
        st.warning("Please upload and clean data before modeling.")
        st.stop()

    # ------------------ اختيار نوع المشكلة ------------------
    problem = st.selectbox(
        "Problem Type",
        ["Classification", "Regression", "Clustering"],
        key="problem_select",
        help="Choose the ML task: Classification (categories), Regression (numbers), Clustering (groups)."
    )

    # ------------------ اختيار العمود الهدف (للـ Supervised فقط) ------------------
    if problem != "Clustering":
        target = st.selectbox(
            "Select target column",
            st.session_state.df.columns.tolist(),
            key="target_select",
            help="The column you want to predict."
        )
    else:
        target = None

    # ------------------ اختيار الموديلات ------------------
    available_models_dict = {
        "Classification": ['LogisticRegression', 'RandomForestClassifier', 'XGBClassifier', 'MLPClassifier', 'KNeighborsClassifier', 'SVC', 'DecisionTreeClassifier'],
        "Regression": ['LinearRegression', 'RandomForestRegressor', 'XGBRegressor', 'MLPRegressor', 'KNeighborsRegressor', 'SVR', 'DecisionTreeRegressor'],
        "Clustering": ['KMeans', 'DBSCAN']
    }

    available_models = available_models_dict.get(problem, [])

    if problem == "Clustering":
        selected_models = st.selectbox("Select Algorithm", available_models, key="algo_select")
    else:
        selected_models = st.multiselect(
            "Select Models to Train",
            available_models,
            default=available_models[:3],
            key="models_select"
        )

    # ------------------ إعدادات التدريب ------------------
    search_type = st.selectbox(
        "Hyperparameter Search Type",
        ["Random", "Grid", "Optuna"],
        key="search_type",
        help="Random & Optuna are faster. Grid is exhaustive."
    )

    n_trials = st.slider("Number of Trials (for Random/Optuna)", 5, 50, 15, key="n_trials")

    confirm_modeling = st.checkbox(
        "Confirm: I understand training may take time (especially with Optuna/Grid).",
        key="confirm_train"
    )

    if confirm_modeling and st.button("Start Modeling", type="primary", width='stretch'):
        if problem != "Clustering" and target is None:
            st.error("Please select a target column for Classification/Regression.")
        else:
            with st.spinner("Training models in progress... This may take 1–3 minutes"):
                progress = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress.progress(i + 1)

                identify_problem(
                    df=st.session_state.df,
                    problem=problem,
                    target=target,
                    selected_models=selected_models,
                    search_type=search_type,
                    n_trials=n_trials
                )
            st.success("Model training completed successfully!")
            st.rerun()

    if problem == "Clustering" and st.session_state.get("training_done"):
        with st.spinner("Preparing data for clustering..."):
            X_train, X_test, _, _, _, _ = prepare_df(st.session_state.df)
            build_clustering_models(X_train, X_test, algorithm=selected_models)

    has_supervised_models = st.session_state.get("best_models_trained") is not None
    has_clustering_model = st.session_state.get("best_clustering_model") is not None

    if has_supervised_models or has_clustering_model:
        st.markdown("---")
        st.success("Training completed successfully! Now save the best model")

        col1, col2 = st.columns([3, 1])
        with col1:
            st.write("**Ready to deploy your model as an API**")
        with col2:
            if st.button("Save & Deploy Model", type="primary", width='stretch', key="save_deploy_btn"):
                st.session_state.show_save_section = True

        # ------------------ قسم الحفظ (يظهر بعد الضغط على الزر) ------------------
        if st.session_state.get("show_save_section", False):
            st.markdown("### Confirm Model Saving")

            # جلب البيانات المطلوبة بأمان
            target_encoder = st.session_state.get("target_encoder_saved")
            feature_names_saved = st.session_state.get("feature_names_saved")
            original_feature_names = st.session_state.get("original_feature_names", [])
            original_dtypes = st.session_state.get("original_dtypes", {})
            problem_type = st.session_state.get("problem_type", "").lower()

            # تحديد أفضل موديل حسب النوع
            if problem_type == "clustering":
                best_model = st.session_state.get("best_clustering_model")
                if not best_model:
                    st.error("Clustering model not found!")
                    st.stop()
                best_name = best_model.__class__.__name__
            else:
                models_dict = st.session_state.get("best_models_trained", {})
                if not models_dict:
                    st.error("No trained models found!")
                    st.stop()

                if problem_type == "regression":
                    priority = ["XGBRegressor", "RandomForestRegressor", "LinearRegression"]
                else:
                    priority = ["XGBClassifier", "RandomForestClassifier", "LogisticRegression"]

                best_name = next((n for n in priority if n in models_dict), list(models_dict.keys())[0])
                best_model = models_dict[best_name]

            st.success(f"**Selected Model:** {best_name}")

            # حفظ الموديل والبيانات المهمة
            if st.button("Confirm Save Model Now", type="primary", width='stretch'):
                import os
                os.makedirs("models", exist_ok=True)

                with st.spinner("Saving model and metadata..."):
                    # حفظ الموديل
                    with open("models/best_model.pkl", "wb") as f:
                        pickle.dump(best_model, f)

                    # حفظ باقي الملفات
                    with open("models/feature_names.pkl", "wb") as f:
                        pickle.dump(feature_names_saved, f)
                    with open("models/original_feature_names.pkl", "wb") as f:
                        pickle.dump(original_feature_names, f)
                    with open("models/original_dtypes.pkl", "wb") as f:
                        pickle.dump(original_dtypes, f)
                    if target_encoder is not None:
                        with open("models/target_encoder.pkl", "wb") as f:
                            pickle.dump(target_encoder, f)

                st.success("All files saved successfully!")
                st.balloons()
                st.info("Go to the **Deployment** tab → Click **Run API**")
                st.markdown("""
                ### Files Saved:
                - `best_model.pkl`
                - `feature_names.pkl`
                - `original_feature_names.pkl`
                - `original_dtypes.pkl`
                - `target_encoder.pkl` (if applicable)
                """)


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