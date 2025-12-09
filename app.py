# ---------------------------------
# Sidebar Navigation (Commented out, using tabs instead)
# ---------------------------------
# st.sidebar.title("App Journey")
# page = st.sidebar.radio("Go To", ["🏠 Home", "🧹 Data Cleaning", "📊 EDA", "🤖 Modeling", "🚀 Deployment"])

import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA
import streamlit as st
from pipeline.data import load_data, clean_data
from pipeline.eda import perform_eda
from pipeline.modeling import identify_problem, build_clustering_models
import subprocess
import time
import os
import io
import pickle
import sys
import zipfile
from io import BytesIO
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
    st.header("🤖 AutoML Modeling")
    st.markdown(
        "Select problem type, configure, train models, view results, and save the best for deployment.")
    if st.session_state.cleaned_df is not None:
        df = st.session_state.cleaned_df
        problem = st.selectbox("Select Problem Type", [
                               "Classification", "Regression", "Clustering"])
        st.session_state.problem_type = problem.lower()

        if problem != "Clustering":
            # Supervised: Classification or Regression
            # Filter useful targets
            target_cols = [col for col in df.columns if df[col].nunique() > 1]
            target = st.selectbox("Select Target Column", target_cols)
            if target:
                if problem == "Classification" and not (df[target].dtype == 'object' or df[target].nunique() < 10):
                    st.warning(
                        "Tip: Target should ideally be categorical for classification.")
                elif problem == "Regression" and not np.issubdtype(df[target].dtype, np.number):
                    st.warning("Tip: Target should be numeric for regression.")

                # Model selection based on problem type (from your modeling.py)
                available_models = {
                    "Classification": ['LogisticRegression', 'RandomForestClassifier', 'XGBClassifier', 'MLPClassifier', 'KNeighborsClassifier', 'SVC', 'DecisionTreeClassifier'],
                    "Regression": ['LinearRegression', 'RandomForestRegressor', 'XGBRegressor', 'MLPRegressor', 'KNeighborsRegressor', 'SVR', 'DecisionTreeRegressor']
                }.get(problem, [])
                selected_models = st.multiselect(
                    "Select Models to Train", available_models, default=available_models[:3])

                search_type = st.selectbox("Hyperparameter Tuning Method", [
                                           "Random", "Grid", "Optuna"])
                n_trials = st.slider(
                    "Number of Trials (for Random/Optuna)", min_value=5, max_value=50, value=10)

                if st.button("🛠️ Train Models", type="primary"):
                    with st.spinner("Training and tuning models..."):
                        identify_problem(df, problem, target,
                                         selected_models, search_type, n_trials)
                    st.success(
                        "Training completed! Check below for metrics and best params.")

                # Display evaluation results, best params, and select/save best model
                if "best_models_trained" in st.session_state and st.session_state.best_models_trained:
                    st.subheader("Model Evaluation Results")
                    st.dataframe(st.session_state.model_evaluation_results)

                    st.subheader("Best Hyperparameters for Each Model")
                    for name, model in st.session_state.best_models_trained.items():
                        with st.expander(f"{name} Best Params"):
                            # Displays tuned params
                            st.json(model.get_params())

                    best_name = st.selectbox(
                        "Select Best Model to Save/Deploy", list(st.session_state.best_models_trained.keys()))
                    if st.button("💾 Save Selected Model as Best"):
                        best_model = st.session_state.best_models_trained[best_name]
                        os.makedirs("models", exist_ok=True)
                        with open("models/best_model.pkl", "wb") as f:
                            pickle.dump(best_model, f)
                        with open("models/feature_names.pkl", "wb") as f:
                            pickle.dump(
                                st.session_state.feature_names_saved, f)
                        if st.session_state.target_encoder_saved:
                            with open("models/target_encoder.pkl", "wb") as f:
                                pickle.dump(
                                    st.session_state.target_encoder_saved, f)
                        st.success(
                            f"{best_name} saved! Proceed to Deployment tab.")
                        st.balloons()

        else:
            # Clustering
            # Import from your modeling.py
            from pipeline.modeling import prepare_df, compute_elbow, KMeans, DBSCAN, silhouette_score
            algorithm = st.selectbox(
                "Select Clustering Algorithm", ["KMeans", "DBSCAN"])
            X_train, X_test, _, _, _, _ = prepare_df(df)  # No target
            X_full = pd.concat([X_train, X_test]).reset_index(drop=True)

            if algorithm == "KMeans":
                st.subheader("KMeans Clustering")
                min_k, max_k = st.slider(
                    "Elbow Method K Range", 2, 15, (2, 10))
                if st.button("📈 Compute Elbow Method"):
                    K, inertias = compute_elbow(X_full, min_k, max_k)
                    fig = px.line(x=K, y=inertias, markers=True,
                                  title="Elbow Method for Optimal K")
                    fig.update_layout(
                        xaxis_title="Number of Clusters (K)", yaxis_title="Inertia")
                    st.plotly_chart(fig)

                n_clusters = st.slider(
                    "Select K (Number of Clusters)", 2, 10, 3)
                if st.button("🛠️ Train KMeans", type="primary"):
                    with st.spinner(f"Training KMeans with K={n_clusters}..."):
                        model = KMeans(n_clusters=n_clusters,
                                       random_state=42, n_init=10)
                        labels = model.fit_predict(X_full)
                        score = silhouette_score(
                            X_full, labels) if n_clusters > 1 else 0
                        st.session_state.best_clustering_model = model
                        st.session_state.clustering_labels = labels
                        st.success(f"Trained! Silhouette Score: {score:.4f}")
                        n_groups = len(set(labels))
                        st.info(f"Data divided into {n_groups} groups.")

                    # 2D Visualization
                    pca = PCA(n_components=2)
                    X_2d = pca.fit_transform(X_full)
                    fig = px.scatter(x=X_2d[:, 0], y=X_2d[:, 1], color=labels.astype(str),
                                     title=f"KMeans Clusters in 2D (K={n_clusters})")
                    st.plotly_chart(fig)

            elif algorithm == "DBSCAN":
                st.subheader("DBSCAN Clustering")
                eps = st.slider("EPS (Radius)", 0.1, 10.0, 0.5, step=0.1)
                min_samples = st.slider("Min Samples", 2, 20, 5)
                if st.button("🛠️ Run DBSCAN", type="primary"):
                    with st.spinner("Running DBSCAN..."):
                        model = DBSCAN(eps=eps, min_samples=min_samples)
                        labels = model.fit_predict(X_full)
                        n_clusters = len(set(labels)) - \
                            (1 if -1 in labels else 0)
                        n_noise = list(labels).count(-1)
                        st.session_state.best_clustering_model = model
                        st.session_state.clustering_labels = labels
                        if n_clusters == 0:
                            st.error("No clusters found! Adjust parameters.")
                        else:
                            score = silhouette_score(
                                X_full, labels) if n_clusters > 1 else -1
                            st.success(
                                f"Found {n_clusters} clusters and {n_noise} noise points. Silhouette Score: {score:.4f}")
                            st.info(f"Data divided into {n_clusters} groups.")

                    # 2D Visualization
                    pca = PCA(n_components=2)
                    X_2d = pca.fit_transform(X_full)
                    fig = px.scatter(x=X_2d[:, 0], y=X_2d[:, 1], color=labels.astype(str),
                                     title="DBSCAN Clusters in 2D")
                    st.plotly_chart(fig)

            # Save best clustering model (unified with supervised saving)
            if "best_clustering_model" in st.session_state:
                if st.button("💾 Save Clustering Model as Best"):
                    os.makedirs("models", exist_ok=True)
                    with open("models/best_model.pkl", "wb") as f:
                        pickle.dump(st.session_state.best_clustering_model, f)
                    with open("models/feature_names.pkl", "wb") as f:
                        pickle.dump(st.session_state.feature_names_saved, f)
                    # No target encoder for clustering
                    if os.path.exists("models/target_encoder.pkl"):
                        os.remove("models/target_encoder.pkl")
                    st.success(
                        "Clustering model saved! Proceed to Deployment tab.")
                    st.balloons()
    else:
        st.warning("⚠️ Please clean data first.")

# ---------------------------------
# Deployment + Inference + Download API
# ---------------------------------
with tab5:
    st.header("Deploy Model as REST API")

    if not os.path.exists("models/best_model.pkl"):
        st.warning(
            "No trained model found. Please complete the Modeling step first.")
    else:
        # Start / Stop API
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Start API Locally", type="secondary"):
                process = subprocess.Popen([
                    sys.executable, "-m", "uvicorn", "pipeline.api:app",
                    "--host", "127.0.0.1", "--port", "8000", "--reload"
                ])
                st.session_state.api_process = process
                time.sleep(3)
                st.success("API is LIVE!")
                st.code("http://127.0.0.1:8000/docs", language="text")

        with col2:
            if st.session_state.get("api_process") and st.button("Stop API", type="secondary"):
                st.session_state.api_process.kill()
                st.session_state.api_process = None
                st.success("API Stopped Successfully")

        st.subheader("Test Model Inference")
        st.markdown("Enter feature values below and get predictions instantly.")

        has_original_features = "original_feature_names" in st.session_state and st.session_state.original_feature_names
        has_cleaned_df = "cleaned_df" in st.session_state and st.session_state.cleaned_df is not None
        has_feature_names_saved = "feature_names_saved" in st.session_state and st.session_state.feature_names_saved

        if has_original_features and has_cleaned_df and has_feature_names_saved:
            inputs = {}
            df_clean = st.session_state.cleaned_df

            for col in st.session_state.original_feature_names:
                if col not in df_clean.columns:
                    continue
                if np.issubdtype(df_clean[col].dtype, np.number):
                    inputs[col] = st.number_input(
                        col, value=0.0, key=f"in_num_{col}")
                else:
                    inputs[col] = st.text_input(
                        col, placeholder="Enter value", key=f"in_text_{col}")

            if st.button("Predict", type="primary"):
                if not st.session_state.get("api_process"):
                    st.error(
                        "API is not running. Click 'Start API Locally' first.")
                else:
                    try:
                        df_in = pd.DataFrame([inputs])
                        df_in = pd.get_dummies(df_in, drop_first=True)
                        df_in = df_in.reindex(
                            columns=st.session_state.feature_names_saved, fill_value=0)
                        payload = {"data": [df_in.iloc[0].to_dict()]}

                        response = requests.post(
                            "http://127.0.0.1:8000/predict", json=payload, timeout=10)

                        if response.status_code == 200:
                            pred = response.json()["predictions"][0]
                            if st.session_state.get("problem_type") == "clustering":
                                st.success(
                                    f"Assigned to **Cluster {int(pred)}**")
                            else:
                                st.success(f"Prediction: **{pred}**")
                            st.balloons()
                        else:
                            st.error(f"API Error: {response.text}")
                    except requests.exceptions.ConnectionError:
                        st.error("Cannot connect to API. Is it running?")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        else:
            st.info(
                "Model inference not available yet. Train and save a model first to enable prediction.")

        st.markdown("---")
        st.subheader("Download Complete API Package")
        st.markdown(
            "One-click download: model + FastAPI + requirements + instructions")

        if st.button("Generate & Download API Package (.zip)", type="primary", use_container_width=True):
            buffer = BytesIO()

            with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                # api.py
                try:
                    with open("pipeline/api.py", "r", encoding="utf-8") as f:
                        zf.writestr("api.py", f.read())
                except FileNotFoundError:
                    zf.writestr("api.py", "# api.py not found")

                # requirements.txt
                zf.writestr("requirements.txt", """fastapi
uvicorn[standard]
pandas
numpy
scikit-learn
xgboost
shap
pydantic
joblib
pickle-mixin""")

                # models folder
                if os.path.exists("models"):
                    for f in os.listdir("models"):
                        path = os.path.join("models", f)
                        if os.path.isfile(path):
                            zf.write(path, f"models/{f}")

                # README
                zf.writestr("README.md", """# ML Forge API - Ready to Deploy!

## Run Locally
pip install -r requirements.txt
uvicorn api:app --reload

Open http://localhost:8000/docs

## Deploy to Cloud
Set command: uvicorn api:app --host 0.0.0.0 --port $PORT
""")

            buffer.seek(0)

            st.download_button(
                label="Download ml_forge_api.zip Now",
                data=buffer,
                file_name="ml_forge_api.zip",
                mime="application/zip",
                type="primary",
                use_container_width=True
            )
            st.success("Package ready! Click above to download")
            st.balloons()