# 🏭 ML Lifecycle Factory Platform

An integrated web platform to automate the **Machine Learning Lifecycle** as a factory-like pipeline.  
This platform streamlines the full Machine Learning Life Cycle (MLLC) from **data ingestion → cleaning → analysis → training → evaluation → deployment as API**.

---

## 📌 Project Idea

The platform aims to simplify the end-to-end machine learning process.  
Users (even non-technical) can upload their datasets, which will automatically pass through all ML lifecycle stages.  
At the end, they will get a **ready-to-use trained model + REST API**.

---

## ⚙️ Project Stages (ML Lifecycle)

1. **Data Upload**

   - Upload CSV, Excel, or JSON files.

2. **Data Cleaning**

   - Handle missing values & duplicates.
   - Remove outliers.
   - Encoding & scaling.

3. **Exploratory Data Analysis (EDA & Visualization)**

   - Descriptive statistics.
   - Interactive visualizations.

4. **Feature Engineering & Preparation**

   - Normalization & Feature scaling.
   - Train/Test Split.

5. **Model Training**

   - Train multiple algorithms (Logistic Regression, Random Forest, XGBoost, Neural Networks).
   - AutoML approach to select the best-performing model.

6. **Evaluation & Optimization**

   - Metrics: Accuracy, Precision, Recall, F1, ROC-AUC.
   - Hyperparameter tuning.
   - Solve ML challenges (Overfitting, Imbalanced Data, etc).

7. **Deployment**
   - Deploy the model as a REST API.
   - Provide a simple **Web UI** (Streamlit).

---

## 🛠️ Technologies Used

- **Python** → Pandas, NumPy, Scikit-learn
- **Deep Learning** → TensorFlow / PyTorch
- **Visualization** → Matplotlib, Seaborn, Plotly
- **AutoML & Tracking** → MLflow, DVC (Git)
- **Deployment** → FastAPI / Flask, Streamlit, Docker
- **Explainable AI (XAI)** → LIME, SHAP

---

## 🚀 Expected Deliverables

- Interactive Web Platform (Streamlit).
- Automated ML pipeline for training and evaluation.
- REST API for model inference.
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

## 📂 Project Structure

```
Automated-ML-Lifecycle/
│── src/
│   ├── backend/
│   │   ├── data.py       # Data upload, cleaning, preparation
│   │   ├── eda.py        # Exploratory analysis & visualization
│   │   ├── train.py      # Model training
│   │   ├── evaluate.py   # Model evaluation
│   │   ├── deploy.py     # Deployment & API
│   └── frontend/
│       └── app.py        # Streamlit Web UI
│
│── visuals/              # Generated charts and visuals
│── requirements.txt
│── README.md
```

---

## 🚦 Getting Started

1. **Clone the Repository**

   ```bash
   git clone https://github.com/Ahmed-7-ML/Automated-Machine-Learning-Life-Cycle.git
   cd Automated-Machine-Learning-Life-Cycle
   ```

2. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**
   ```bash
   streamlit run src/frontend/app.py
   ```

---

✨ Now, upload your dataset and let the platform handle the full **Machine Learning Lifecycle** for you!
