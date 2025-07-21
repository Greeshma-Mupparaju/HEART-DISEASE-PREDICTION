# HEART-DISEASE-PREDICTION
Heart Disease Prediction using Machine Learning
This project aims to predict the likelihood of heart disease in patients using machine learning algorithms. By analyzing medical attributes such as age, cholesterol level, blood pressure, and more, the system helps in early detection and preventive care.

📌 Features
Data preprocessing and feature scaling

Model training using Logistic Regression

Evaluation using accuracy, confusion matrix, and classification report

Comparison with other models (Random Forest, SVM)

Sample prediction for new patient data

📂 Dataset
Dataset used: Heart Disease UCI Dataset

Contains medical features like:

Age

Resting Blood Pressure

Cholesterol

Chest Pain Type

Maximum Heart Rate

Fasting Blood Sugar

ECG Results

ST Depression

Target (Presence of Heart Disease)

🛠️ Tech Stack
Python

NumPy, Pandas

Matplotlib, Seaborn

Scikit-learn

📊 Machine Learning Models Used
Logistic Regression ✅

Random Forest Classifier 🌲

Support Vector Machine (SVM) 🧠

📈 Accuracy
Logistic Regression: ~85% (varies with train/test split)

Random Forest: ~90% (depending on hyperparameters)

💡 Purpose
This project was created as a healthcare-focused machine learning application to showcase:

The power of data in making predictions

The use of supervised learning models for binary classification

Real-world applicability of ML in medical diagnostics

🚀 How to Run
bash
Copy
Edit
# Clone the repository
git clone https://github.com/yourusername/heart-disease-prediction.git
cd heart-disease-prediction

# Install required packages
pip install -r requirements.txt

# Run the script
python heart_disease_prediction.py

# ❤️ Heart Disease Prediction using Machine Learning

This project uses machine learning techniques to predict the likelihood of heart disease based on patient health parameters. It aims to assist medical professionals and individuals in early diagnosis and risk assessment.

## 📊 Overview

Heart disease is one of the leading causes of death globally. Early detection is key to prevention and treatment. This project applies classification algorithms to a real-world medical dataset to predict the presence of heart disease.

---

## 📁 Dataset

- **Source**: [Heart Disease UCI Dataset on Kaggle](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)
- **Features**:
  - Age
  - Sex
  - Chest pain type
  - Resting blood pressure
  - Cholesterol
  - Fasting blood sugar
  - Rest ECG
  - Max heart rate
  - Exercise-induced angina
  - ST depression
  - Slope, Ca, Thal
  - Target (1 = Heart Disease, 0 = No Heart Disease)

---

## 🧠 Machine Learning Models Used

- Logistic Regression ✅
- Random Forest Classifier 🌲
- Support Vector Machine (SVM) 💻

---

## 🛠️ Technologies

- Python
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

---

## 🚀 How to Run the Project

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/heart-disease-prediction.git
   cd heart-disease-prediction

📈 Results
Logistic Regression Accuracy: ~85%

Random Forest Accuracy: ~90%

SVM Accuracy: ~88%

Evaluation Metrics: Confusion Matrix, Classification Report, Accuracy Score

💡 Future Improvements
Hyperparameter tuning for better accuracy

Web app interface using Streamlit or Flask

Model deployment using cloud services

