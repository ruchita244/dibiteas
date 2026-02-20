# DibiTeas – AI Powered Diabetes Prediction & Health Analyzer

DibiTeas is a **Flask web application** designed to predict diabetes risk and analyze basic health parameters.  
It provides a modern, interactive, and **professional UI** with animations and a health dashboard.

---

## 🚀 Features

- **Login / Signup system** using a JSON-based database (`database.json`)  
- **Diabetes Prediction** using 8 medical parameters:
  - Pregnancies  
  - Glucose  
  - BloodPressure  
  - SkinThickness  
  - Insulin  
  - BMI  
  - DiabetesPedigreeFunction  
  - Age
- **Health Analyzer** based on Age, BMI, and Glucose  
- **Professional UI** with animations, gradient backgrounds, and glow effects  
- **Auto-open browser** for convenience  

---

## 📁 Project Structure
DibiTeas/
├─ app.py # Flask backend (ML + Health Analyzer)
├─ diabetes.csv # Sample dataset
├─ requirements.txt # Python dependencies
├─ README.md # This file
└─ templates/
├─ home.html # Landing page
├─ login.html # Login page
├─ signup.html # Signup page
└─ health.html # Health dashboard
