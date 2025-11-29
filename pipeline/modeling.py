# ---------------------------------
# Auto-ML Python Script
# Building Machine Learning Models based on a specific problem (Classification, Regression, Clustering)
# Apply the suitable algorithms for the problem
# Train them and Evaluate based in the specific metrics
# Select the best model for future deploymnet
# ---------------------------------
import numpy as np
import pandas as pd
import streamlit as st

# Modeling Libraries
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report,
    mean_absolute_error as mae, mean_absolute_percentage_error as mape,
    mean_squared_error as mse, r2_score as r2, silhouette_score
)
from sklearn.cluster import KMeans, DBSCAN
from xgboost import XGBRegressor, XGBClassifier
# Imbalanced Data Handling
from imblearn.over_sampling import SMOTE
# Plotly for Elbow Method
import plotly.graph_objects as go

def init_state():
    defaults = {
        "best_models_trained": None,
        "target_encoder_saved": None,
        "feature_names_saved": None,
        "problem_type": None,
        "modeling_running": False,
        "clustering_algorithm": "KMeans",
        "k_value": 3
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ==================== Data Preparation for Modeling ====================
def prepare_df(df, target=None):
    df = df.copy()
    if target not in df.columns:
        raise ValueError("Target column not in DataFrame")

    X = df.drop(columns=[target])
    if target != None:
        y = df[target]

    # Save feature names before encoding
    feature_names = X.columns.tolist()

    # Encode target if categorical
    target_encoder = None
    if y.dtype == 'object' or y.dtype.name == 'category':
        target_encoder = LabelEncoder()
        y = target_encoder.fit_transform(y.astype(str))

    # One-hot encode categorical features
    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y if target_encoder is not None else None
    )

    # Save to session state for deployment
    st.session_state.feature_names_saved = X.columns.tolist()
    st.session_state.target_encoder_saved = target_encoder

    return X_train, X_test, y_train, y_test, target_encoder, feature_names

# ==================== Define the Problem and Target ====================
def identify_problem(df, problem, target=None):
    st.session_state.problem_type = problem.lower()

    if problem.lower() == 'clustering':
        X = pd.get_dummies(df, drop_first=True)
        X_train, X_test = train_test_split(X, test_size=0.3, random_state=42)
        build_clustering_models(X_train, X_test)
        return

    X_train, X_test, y_train, y_test, target_encoder, feature_names = prepare_df(df, target)

    st.write(f"### ⚙️ Preparing Data for **{problem}** Model Training")
    st.write(f"Training set: `{X_train.shape}` | Test set: `{X_test.shape}`")

    if problem.lower() == 'regression':
        best_models = build_regression_models(X_train, y_train)
        evaluate_regression_models(best_models, X_test, y_test)

    elif problem.lower() == 'classification':
        # Check for class imbalance
        unique, counts = np.unique(y_train, return_counts=True)
        imbalance_ratio = counts.max() / counts.min() if len(counts) > 1 else 1
        if len(unique) > 1 and imbalance_ratio > 2.0:
            st.warning(f"⚠️ Detected class imbalance (ratio: {imbalance_ratio:.1f}). Applying SMOTE...")
            smote = SMOTE(random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            st.success("SMOTE applied successfully!")

        best_models = build_classification_models(X_train, y_train)
        evaluate_classification_models(best_models, X_test, y_test)

    # Save best models to session state
    st.session_state.best_models_trained = best_models

# ==================== REGRESSION ====================
def build_regression_models(X_train, y_train):
    best_models = {}

    models = {
        'LinearRegression': LinearRegression(),
        'Ridge': LinearRegression(),  # We'll use Ridge via params later if needed
        'RandomForestRegressor': RandomForestRegressor(random_state=42),
        'XGBRegressor': XGBRegressor(random_state=42),
        'MLPRegressor': MLPRegressor(max_iter=500, random_state=42),
        'KNeighborsRegressor': KNeighborsRegressor(),
        'SVR': SVR(),
        'DecisionTreeRegressor': DecisionTreeRegressor(random_state=42)
    }

    params = {
        'RandomForestRegressor': {'n_estimators': [100, 200, 300], 'max_depth': [10, 20, None]},
        'XGBRegressor': {'n_estimators': [100, 200], 'max_depth': [6, 10], 'learning_rate': [0.01, 0.1]},
        'MLPRegressor': {'hidden_layer_sizes': [(50,), (100,), (50, 50)], 'alpha': [0.0001, 0.001]},
        'KNeighborsRegressor': {'n_neighbors': [3, 5, 7, 11], 'weights': ['uniform', 'distance']},
        'SVR': {'C': [0.1, 1, 10], 'kernel': ['rbf', 'linear']},
        'DecisionTreeRegressor': {'max_depth': [10, 20, None], 'min_samples_split': [2, 5]}
    }

    with st.spinner("Training Regression Models..."):
        for name, model in models.items():
            if model is None:
                continue
            st.info(f"Training **{name}**")
            try:
                if name in ['LinearRegression']:
                    model.fit(X_train, y_train)
                    best_models[name] = model
                else:
                    search = RandomizedSearchCV(
                        model, params.get(name, {}), n_iter=10, cv=5,
                        scoring='r2', n_jobs=-1, random_state=42
                    )
                    search.fit(X_train, y_train)
                    best_models[name] = search.best_estimator_
                    st.write(f"Best R² (CV): {search.best_score_:.3f}")
            except Exception as e:
                st.error(f"{name} failed: {e}")

    st.session_state.best_models_trained = best_models
    return best_models

# ==================== CLASSIFICATION ====================
def build_classification_models(X_train, y_train):
    best_models = {}

    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000),
        'RandomForestClassifier': RandomForestClassifier(random_state=42),
        'XGBClassifier': XGBClassifier(random_state=42),
        'MLPClassifier': MLPClassifier(max_iter=500, random_state=42),
        'KNeighborsClassifier': KNeighborsClassifier(),
        'SVC': SVC(probability=True),
        'DecisionTreeClassifier': DecisionTreeClassifier(random_state=42)
    }

    params = {
        'RandomForestClassifier': {'n_estimators': [100, 200, 300], 'max_depth': [10, 20, None]},
        'XGBClassifier': {'n_estimators': [100, 200], 'max_depth': [6, 10], 'learning_rate': [0.01, 0.1]},
        'MLPClassifier': {'hidden_layer_sizes': [(50,), (100,), (100, 50)], 'alpha': [0.0001, 0.001]},
        'KNeighborsClassifier': {'n_neighbors': [3, 5, 7, 11], 'weights': ['uniform', 'distance']},
        'SVC': {'C': [0.1, 1, 10], 'kernel': ['rbf', 'linear']},
        'DecisionTreeClassifier': {'max_depth': [10, 20, None], 'min_samples_split': [2, 5]}
    }

    with st.spinner("Training Classification Models..."):
        for name, model in models.items():
            if model is None:
                continue
            st.info(f"Training **{name}**")
            try:
                search = RandomizedSearchCV(
                    model, params.get(name, {}), n_iter=10, cv=5,
                    scoring='f1_weighted', n_jobs=-1, random_state=42
                )
                search.fit(X_train, y_train)
                best_models[name] = search.best_estimator_
                st.write(f"Best F1 (CV): {search.best_score_:.3f}")
            except Exception as e:
                st.error(f"{name} failed: {e}")

    st.session_state.best_models_trained = best_models
    return best_models

# ==================== EVALUATION WITH CROSS-VAL ====================
def evaluate_regression_models(best_models, X_test, y_test):
    st.header("Regression Models Evaluation")
    results = []

    for name, model in best_models.items():
        with st.expander(f"**{name}**"):
            y_pred = model.predict(X_test)

            # Cross-validation score
            cv_scores = cross_val_score(model, X_test, y_test, cv=5, scoring='r2')
            st.write(f"**Cross-Val R²:** `{cv_scores.mean():.3f} ± {cv_scores.std():.3f}`")

            r2_test = r2(y_test, y_pred)
            results.append({'Model': name, 'R² Test': r2_test, 'CV R²': cv_scores.mean()})

            st.write(f"**R² (Test):** `{r2_test:.3f}`")
            st.write(f"**MAE:** `{mae(y_test, y_pred):.3f}`")
            st.write(f"**MAPE:** `{mape(y_test, y_pred):.3f}`")
            st.write(f"**MSE:** `{mse(y_test, y_pred):.3f}`")
            st.write(f"**RMSE:** `{np.sqrt(mse(y_test, y_pred)):.3f}`")

    # Summary table
    results_df = pd.DataFrame(results).sort_values(by='R² Test', ascending=False)
    st.write("### Best Model Summary")
    st.dataframe(results_df, use_container_width=True)

def evaluate_classification_models(best_models, X_test, y_test):
    st.header("Classification Models Evaluation")
    results = []

    for name, model in best_models.items():
        with st.expander(f"**{name}**"):
            y_pred = model.predict(X_test)

            # Cross-validation
            cv_scores = cross_val_score(model, X_test, y_test, cv=5, scoring='f1_weighted')
            st.write(f"**Cross-Val F1:** `{cv_scores.mean():.3f} ± {cv_scores.std():.3f}`")

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            results.append({'Model': name, 'Accuracy': acc, 'F1': f1, 'CV F1': cv_scores.mean()})

            st.write(f"**Accuracy:** `{acc:.3f}` | **F1-Score:** `{f1:.3f}`")
            st.text(classification_report(y_test, y_pred))
            st.write("**Confusion Matrix:**")
            st.dataframe(pd.DataFrame(confusion_matrix(y_test, y_pred)), use_container_width=True)

    results_df = pd.DataFrame(results).sort_values(by='F1', ascending=False)
    st.write("### Best Model Summary")
    st.dataframe(results_df, use_container_width=True)

# ==================== CLUSTERING WITH ELBOW METHOD ====================
def build_clustering_models(X_train, X_test):
    st.subheader("Unsupervised Clustering")
    algorithm = st.selectbox("Select clustering algorithm", ["KMeans", "DBSCAN"], help='Clustering Algorithm')

    X = pd.concat([X_train, X_test])
    best_model = None
    best_score = -1

    if algorithm == 'kmeans':
        st.write("### 📌 Using KMeans Clustering")
        # Elbow Method
        inertias = []
        K = range(2, 11)
        for k in K:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(K), y=inertias, mode='lines+markers'))
        fig.update_layout(title="Elbow Method for Optimal K", xaxis_title="Number of Clusters", yaxis_title="Inertia")
        st.plotly_chart(fig, use_container_width=True)

        # User selects K
        n_clusters = st.slider("Select number of clusters", 2, 10, 3)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(X)

        score = silhouette_score(X, labels) if n_clusters > 1 else None
        st.write(f"**Silhouette Score:** `{score:.3f}`" if score else "N/A")
        st.write("Cluster centers shape:", kmeans.cluster_centers_.shape)

        best_model = kmeans
        best_score = score
    
    elif algorithm == 'DBSCAN':
        st.write("### 📌 Using DBSCAN")

        eps = st.slider("eps (neighborhood radius)", 0.1, 5.0, 0.5)
        min_samples = st.slider("min_samples", 3, 20, 5)

        db = DBSCAN(eps=eps, min_samples=min_samples)
        labels = db.fit_predict(X)

        # If all points become -1 -> invalid
        if len(set(labels)) <= 1:
            st.error("⚠️ DBSCAN failed to form clusters — adjust parameters.")
            return None

        score = silhouette_score(X, labels)
        st.success(f"**Silhouette Score (DBSCAN):** `{score:.4f}`")
        st.write(f"Clusters found: {len(set(labels))}")

        best_model = db
        best_score = score

    # Save the model inside session
    st.session_state["best_clustering_model"] = best_model
    st.session_state["best_clustering_score"] = best_score

    return best_model
