# 🏭 ML Lifecycle Factory Platform

An integrated web platform to automate the **Machine Learning Lifecycle** as a factory-like pipeline.
A simple platform to automate the Machine Learning Life Cycle (MLLC) from data ingestion to API deployment.

---

## 📌 Project Idea

The platform aims to simplify the end-to-end machine learning process. Users (even non-technical) can upload their datasets, which will automatically pass through all ML lifecycle stages (Cleaning → Exploration → Training → Evaluation → Deployment).  
At the end, they will get a **ready-to-use model + REST API**.

---

## ⚙️ Project Stages (ML Lifecycle)

1. **Data Upload**

   - Upload CSV/Excel/JSON files.

2. **Data Cleaning**

   - Handle missing values & duplicates.
   - Remove outliers.
   - Encoding & Scaling.

3. **Exploratory Data Analysis (EDA & Visualization)**

   - Descriptive statistics.
   - Visualizations .

4. **Feature Engineering & Preparation**

   - Normalization & Feature Scaling.
   - Train/Test Split.

5. **Model Training**

   - Train multiple algorithms (eg. Logistic Regression, Random Forest, XGBoost, Neural Networks).
   - AutoML approach to select the best-performing model.

6. **Evaluation & Optimization**

   - Metrics (Accuracy, Precision, Recall, F1, ROC-AUC, .....).
   - Hyperparameter Tuning.
   - Solve ML challenges (Overfitting, Imbalanced Data, ....).

7. **Deployment**
   - Deploy the model as API .
   - Simple Web UI (Streamlit/React or any).

---

## 🛠️ Technologies Used

- **Python**: Pandas, NumPy, Scikit-learn.
- **Deep Learning**: TensorFlow / PyTorch.
- **Visualization**: Matplotlib, Seaborn, Plotly.
- **AutoML & Experiment Tracking**: MLflow, DVC (Git).
- **Deployment**: FastAPI / Flask, Streamlit, Docker.
- **Explainable AI**: LIME, SHAP.

---

## 🚀 Expected Deliverables

- Interactive Web Platform.
- Automated ML pipeline for training and evaluation.
- API for model usage.
- Analytical reports & Explainable AI visualizations.

---

## 👨‍💻 Team Members

- **Ahmed Akram Amer** (Team Leader)
- **Eyad Sherif Rashad**
- **Ziad Moataz Hawana**
- **Mohamed Adel Tawfik**
- **Ahmed Mohamed Abdel-Mordi**
- **Salem Mohamed El-Katatny**

### Under the supervision of Eng. **Mostafa Sami Atlam**

---

## Structure

- `src/backend/`: Handles data processing and model training.
- `src/frontend/`: Manages the user interface.
- `visuals/`: Stores generated charts and visuals.

Automated-ML-Lifecycle/
│── src/
│ ├── backend/
│ │ ├── data.py # تحميل وتنظيف وتجهيز البيانات
│ │ ├── eda.py # التحليل و Visualization
│ │ ├── train.py # تدريب النماذج
│ │ ├── evaluate.py # تقييم النماذج
│ │ ├── deploy.py # تخزين و API
│ └── frontend/
│ └── app.py # Streamlit واجهة المستخدم
│── requirements.txt
│── README.md

## Getting Started

- Clone: `git clone https://github.com/Ahmed-7-ML/Automated-Machine-Learning-Life-Cycle.git`
- Install: `pip install -r requirements.txt`
- Run: `streamlit run src/frontend/app.py`
