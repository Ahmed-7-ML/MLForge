# ====================================
# Auto-ML Python Script
# Building Machine Learning Models based on a specific problem (Classification, Regression, Clustering)
# Apply the suitable algorithms for the problem
# Train them and Evaluate based in the specific metrics
# Select the best model for future deploymnet
# ====================================

import numpy as np 
import pandas as pd

# For integrate this script with the app
import streamlit as st

# Helper Functions
from pipeline.data import clean_data

# Modeling Libraries (Regression, Classification, Clustering)
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.cluster import KMeans, DBSCAN

# Preprocessing Libraries
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
# Metrics Library
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.metrics import mean_absolute_error as mae, mean_absolute_percentage_error as mape, mean_squared_error as mse, r2_score as r2
from sklearn.metrics import silhouette_score

def prepare_df(df, target):
    """
    Splits dataset into Features (X) and Target (y) and then into Training & Testing Sets
    Handles categorical target by encoding it.
    """
    df = df.copy()
    if target not in df.columns:
        raise ValueError("Target column not in DataFrame")
    
    X = df.drop(columns=[target])
    y = df[target]

    # Encode target if categorical (strings/object)
    target_encoder = None
    if y.dtype == object or y.dtype.name == 'category':
        target_encoder = LabelEncoder()
        y = target_encoder.fit_transform(y.astype(str))

    # One-hot encode X for object columns
    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test, target_encoder

def identify_problem(df, target, problem):
    """
    Identifies the problem type and calls the appropriate model building function.
    """

    X_train, X_test, y_train, y_test, target_encoder = prepare_df(df, target)
    st.write(f"### ⚙️ Preparing Data for {problem} Model Training")
    st.write(f"Training set shape: {X_train.shape}")
    st.write(f"Testing set shape: {X_test.shape}")

    if problem.lower() == 'regression':
        best_models = build_regression_models(X_train, y_train)
        evaluate_regression_models(best_models, X_test, y_test)

    elif problem.lower() == 'classification':
        best_models = build_classification_models(X_train, y_train)
        evaluate_classification_models(best_models, X_test, y_test)

    elif problem.lower() == 'clustering':
        build_clustering_models(X_train, X_test)

def build_regression_models(X_train, y_train):
    """
    Try Different ML Models for Regression Only
    Apply Grid Search to find the Best Estimators
    Args:
        X_train: Training Data (Features Only)
        y_train: Training Labels
    Returns:
        The Best Models of Regression
    """

    best_models = {}

    models = {
        'LinearRegression': LinearRegression(),
        'SVR':SVR(),
        'KNeighborsRegressor': KNeighborsRegressor(),
        'DecisionTreeRegressor': DecisionTreeRegressor(),
        'RandomForestRegressor': RandomForestRegressor()
    }
    params = {
        'RandomForestRegressor': {
            'n_estimators': [50, 100, 150, 200],
            'max_depth': [10, 15, 20],
            'min_samples_split': [3, 5, 7, 9]
        },
        'KNeighborsRegressor': {
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance'],
            'p':[1, 2]
        },
        'DecisionTreeRegressor':
        {
            'max_depth': [10, 15, 20],
            'min_samples_split': [3, 5, 7, 9]
        },
        'SVR': {
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'degree': [2, 3, 4],
            'gamma': ['scale', 'auto'],
            'C': [0.1, 1, 10, 100]
        }
    }
    with st.spinner("⚙️ Training Regression Models... Please wait"):
        for name, model in models.items():
            st.info(f"Training **{name}**.......")
            if name == 'LinearRegression':
                model.fit(X_train, y_train)
                best_models[name] = model
                # st.success(f"Trained **{name}**")
                continue
            try:
                search = RandomizedSearchCV(model, param_distributions=params.get(name, {}), cv=5, n_jobs=-1, n_iter=10, verbose=1, scoring='r2', random_state=42)
                search.fit(X_train, y_train)
                best_models[name] = search.best_estimator_
                st.write(f"✅ **{name}** Best Parameters: `{search.best_params_}`")
                st.write(f"⭐ **{name}** Best R2 Score: `{round(search.best_score_, 3)}`")
                st.markdown('---')
            except Exception as e:
                st.error(f"Failed to Train {name}: {e}")

    return best_models

def build_classification_models(X_train, y_train):
    """
    Try Different ML Models for Classification Only
    Apply Grid Search to find the Best Estimators
    Args:
        X_train: Training Data (Features Only)
        y_train: Training Labels
    Returns:
        The Best Models of Classification
    """

    best_models = {}
    models = {
        'LogisticRegression': LogisticRegression(),
        'SVC':SVC(),
        'KNeighborsClassifier': KNeighborsClassifier(),
        'DecisionTreeClassifier': DecisionTreeClassifier(),
        'RandomForestClassifier': RandomForestClassifier()
    }

    params = {
        'LogisticRegression': {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga']
        },
        'RandomForestClassifier': {
            'n_estimators': [50, 100, 150, 200],
            'max_depth': [10, 15, 20],
            'min_samples_split': [3, 5, 7, 9]
        },
        'KNeighborsClassifier': {
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance'],
            'p':[1, 2]
        },
        'DecisionTreeClassifier':
        {
            'max_depth': [10, 15, 20],
            'min_samples_split': [3, 5, 7, 9]
        },
        'SVC': {
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'degree': [2, 3, 4],
            'gamma': ['scale', 'auto'],
            'C': [0.1, 1, 10, 100]
        }
    }
    with st.spinner("⚙️ Training Classification Models... Please wait"):
        for name, model in models.items():
            st.info(f"Training {name}.......")
            try:
                search = RandomizedSearchCV(model, param_distributions=params[name], cv=5, n_jobs=-1, verbose=1, n_iter=10, scoring='accuracy', random_state=42)
                search.fit(X_train, y_train)
                best_models[name] = search.best_estimator_
                st.write(f"Best Parameters\n{search.best_params_}")
                st.write(f"Best Accuracy = {round(search.best_score_, 3)}")
                st.markdown('---')
            except Exception as e:
                st.error(f"Failed to Train {name}: {e}")

    return best_models

def build_clustering_models(X_train, X_test):
    st.info("Running simple clustering (KMeans) over features (ignores target).")
    X = pd.concat([X_train, X_test], axis=0)
    try:
        n_clusters = st.number_input("Number of clusters (KMeans)", min_value=2, max_value=20, value=3)
        km = KMeans(n_clusters=n_clusters, random_state=42)
        labels = km.fit_predict(X)
        score = silhouette_score(X, labels) if len(np.unique(labels)) > 1 else None
        st.write(f"KMeans Silhouette Score: {round(score, 3) if score is not None else 'N/A'}")
        st.write("Cluster centers shape:", km.cluster_centers_.shape)
    except Exception as e:
        st.error(f"Clustering error: {e}")

def evaluate_regression_models(best_models, X_test, y_test):
    """
    Evaluates the trained regression models on the test set and displays metrics.
    """
    st.header("📈 Models Evaluation")
    with st.spinner("⚙️ Evaluating Regression Models... Please wait"):
        for name, model in best_models.items():
            st.subheader(f"Evaluating **{name}**....")
            y_pred = model.predict(X_test)

            with st.expander("Show Metrics"):
                st.write("#### 📊 Metrics")
                st.write(f"**Mean Absolute Error (MAE):** `{round(mae(y_test, y_pred), 3)}`")
                st.write(f"**Mean Absolute Percentage Error (MAPE):** `{round(mape(y_test, y_pred), 3)}`")
                st.write(f"**Mean Squared Error (MSE):** `{round(mse(y_test, y_pred), 3)}`")
                st.write(f"**Root Mean Squared Error (RMSE):** `{round(np.sqrt(mse(y_test, y_pred)), 3)}`")
                st.write(f"**R² Score:** `{round(r2(y_test, y_pred), 3)}`")
            st.markdown('---')
    st.success("🎉 All Regression models trained successfully!")

def evaluate_classification_models(best_models, X_test, y_test):
    """
    Evaluates the trained classification models on the test set and displays metrics.
    """
    st.header("📈 Model Evaluation")
    with st.spinner("⚙️ Evaluating Classification Models... Please wait"):
        for name, model in best_models.items():
            st.subheader(f"Evaluating **{name}**....")
            y_pred = model.predict(X_test)
            with st.expander("Show Metrics"):
                st.write("#### 📊 Metrics")
                st.write(f"**Accuracy:** `{round(accuracy_score(y_test, y_pred), 3)}`")
                st.write(f"**Classification Report:**")
                st.text(classification_report(y_test, y_pred))
                st.write(f"**Confusion Matrix:**")
                st.dataframe(pd.DataFrame(confusion_matrix(y_test, y_pred)), use_container_width=True)
            st.markdown('---')
    st.success("🎉 All Classification models trained successfully!")