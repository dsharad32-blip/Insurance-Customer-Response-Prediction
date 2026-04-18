# 📊 Insurance Customer Response Prediction using Machine Learning

This project predicts whether a customer will respond to a vehicle insurance offer.  
The model is trained using classical machine learning algorithms and is ready to be deployed as an interactive application.

## Problem Statement
Predicting customer response helps insurance companies:
- 🔹 Retain high-value customers
- 🔹 Increase conversion of offers
- 🔹 Design targeted marketing campaigns

This project predicts whether a customer is likely to respond based on demographics, vehicle information, and policy details.

## Key Features
- 🔹 End-to-end ML pipeline
- 🔹 Data cleaning & exploratory analysis using Jupyter Notebooks
- 🔹 Feature engineering & preprocessing
- 🔹 Multiple ML models trained and evaluated
- 🔹 Best-performing model saved for deployment
- 🔹 Production-ready project structure

## Machine Learning Workflow
Raw Customer Data (`data.csv`)  
↓  
Data Cleaning & Preprocessing (`preprocessing.ipynb`)  
↓  
Exploratory Data Analysis (`EDA.ipynb`)  
↓  
Model Training & Evaluation (`training.py`)  
↓  
Best Model Selection  
↓  
Ready for Deployment


## 📁 Project Structure
Insurance_Customer_Response_Prediction/
│
├── app.py                   # Streamlit entry point (not yet deployed)
├── requirements.txt         # Project dependencies
├── README.md
├── .gitignore               # Git ignore file
│
├── DATA/                    # Data folder
│   ├── data.csv             # Raw dataset
│   └── clean_data.csv       # Cleaned dataset
│
├── Models/                   # Trained model & metrics
│   ├── Best_model.pkl       # Best model
│   └── model_results.csv    # Model comparison metrics
│
├── Notebooks/               # Analysis & preprocessing
│   ├── EDA.ipynb            # Exploratory data analysis
│   └── preprocessing.ipynb  # Data preprocessing
│
├── SRC/                     # Core ML logic
│   └── training.py          # Training pipeline
│
└── venv/                    # Virtual environment (ignored)

## 📊 Dataset
- 🔹 **data.csv** – Raw dataset with customer, vehicle, and policy information
- 🔹 **clean_data.csv** – Cleaned dataset generated after preprocessing

Dataset contains features like:
- 🔹 Age, Gender, Region
- 🔹 Vehicle Age, Vehicle Damage
- 🔹 Policy details, Annual Premium
- 🔹 Historical customer behavior

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository
```bash
git clone <your-repo-url>
cd Insurance_Customer_Response_Prediction
```
### 2️⃣ Create & activate virtual environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3️⃣ Install dependencies
pip install -r requirements.txt

## 🧹 Data Preparation
- 🔹 Preprocessing done via preprocessing.ipynb
- 🔹 Cleaned dataset exported to DATA/clean_data.csv
- 🔹 EDA performed in EDA.ipynb for feature insight

## 🤖 Model Training
- 🔹 Run the training pipeline:
python SRC/training.py

- 🔹 Loads cleaned data
- 🔹 Splits train/test sets
- 🔹 Builds preprocessing & ML pipelines
- 🔹 Trains multiple models and evaluates metrics

- 🔹 Saves:
  - Model/Best_model.pkl
  - Model/model_results.csv

## 📊 Model Evaluation
- 🔹 The saved model_results.csv contains:
  - Model names
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - ROC-AUC

## 📦 Deployment (Future)
- 🔹 Once app.py is ready, the project can be deployed using Streamlit.

## 📊 Technologies Used
- 🔹 Python 3.10
- 🔹 Pandas
- 🔹 NumPy
- 🔹 Scikit-Learn
- 🔹 Matplotlib
- 🔹 Seaborn
- 🔹 Streamlit
- 🔹 Joblib
- 🔹 Jupyter Notebook

## 🚀 Future Enhancements
- 🔹 Streamlit-based UI for predictions
- 🔹 Batch prediction support
- 🔹 Explainable AI (SHAP)
- 🔹 FastAPI REST API
- 🔹 Docker deployment
- 🔹 Monitoring and model drift detection

## 🤝 Contributing
- 🔹 Contributions, suggestions, and improvements are welcome
- 🔹 Feel free to open an issue or submit a pull request

## 🙏 Acknowledgements
- 🔹 Inspired by insurance customer analytics datasets
- 🔹 Thanks to the open-source Python ecosystem