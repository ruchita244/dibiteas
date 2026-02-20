import pandas as pd
import numpy as np
import json
import os
from flask import Flask, request, jsonify, render_template
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import webbrowser
import threading

app = Flask(__name__)

# ---------------------------
#  MACHINE LEARNING MODEL (8 COLUMNS ONLY)
# ---------------------------
df = pd.read_csv("diabetes.csv")

# EXACT 8 MODEL INPUT COLUMNS
ML_COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age"
]

X = df[ML_COLUMNS]
y = df["Outcome"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# ---------------------------
# USER DATABASE (JSON FILE)
# ---------------------------
DB_FILE = "database.json"

# Create file if not exists
if not os.path.exists(DB_FILE):
    with open(DB_FILE, "w") as f:
        json.dump({"users": {}}, f)


def load_db():
    with open(DB_FILE, "r") as f:
        return json.load(f)


def save_db(data):
    with open(DB_FILE, "w") as f:
        json.dump(data, f, indent=4)


# ---------------------------
# FRONTEND ROUTES
# ---------------------------
@app.route("/")
def home():
    return render_template("home.html")


@app.route("/signup")
def signup():
    return render_template("signup.html")


@app.route("/login")
def login():
    return render_template("login.html")


@app.route("/health")
def health():
    return render_template("health.html")


# ---------------------------
# SIGNUP API
# ---------------------------
@app.post("/api/signup")
def api_signup():
    data = request.json
    db = load_db()

    email = data["email"]
    if email in db["users"]:
        return jsonify({"status": "error", "message": "User already exists!"})

    db["users"][email] = {
        "name": data["name"],
        "password": data["password"]
    }

    save_db(db)
    return jsonify({"status": "success", "message": "Signup successful!"})


# ---------------------------
# LOGIN API
# ---------------------------
@app.post("/api/login")
def api_login():
    data = request.json
    db = load_db()

    email = data["email"]
    password = data["password"]

    if email not in db["users"]:
        return jsonify({"status": "error", "message": "User not found"})

    if db["users"][email]["password"] != password:
        return jsonify({"status": "error", "message": "Incorrect password"})

    return jsonify({
        "status": "success",
        "message": f"Welcome {db['users'][email]['name']}!"
    })


# ---------------------------
# ML DIABETES PREDICTOR (8 Inputs)
# ---------------------------
@app.post("/predict")
def predict():
    data = request.json

    try:
        values = np.array([float(data[col]) for col in ML_COLUMNS]).reshape(1, -1)
    except:
        return jsonify({"error": "Missing or invalid input values"}), 400

    scaled = scaler.transform(values)

    pred = model.predict(scaled)[0]
    prob = model.predict_proba(scaled)[0][1]

    return jsonify({
        "prediction": int(pred),
        "probability": round(float(prob), 3)
    })


# ---------------------------
# HEALTH ANALYZER FEATURE ❤️
# ---------------------------
@app.post("/analyze")
def analyze_health():
    data = request.json

    age = float(data["age"])
    bmi = float(data["bmi"])
    glucose = float(data["glucose"])

    score = 0

    # AGE
    if age < 30:
        score += 0
    elif 30 <= age < 45:
        score += 1
    elif 45 <= age < 60:
        score += 2
    else:
        score += 3

    # BMI
    if bmi < 18.5:
        score += 0
    elif 18.5 <= bmi < 25:
        score += 0
    elif 25 <= bmi < 30:
        score += 1
    elif 30 <= bmi < 35:
        score += 2
    else:
        score += 3

    # GLUCOSE
    if glucose < 110:
        score += 0
    elif 110 <= glucose < 140:
        score += 1
    elif 140 <= glucose < 180:
        score += 2
    else:
        score += 3

    # FINAL RISK CATEGORY
    if score <= 2:
        risk = "Low"
        color = "#00ff88"
        tips = "Fantastic! Maintain your healthy lifestyle 😊"
    elif 3 <= score <= 5:
        risk = "Medium"
        color = "#ffd000"
        tips = "Improve diet, hydrate well, and walk 30 minutes daily."
    else:
        risk = "High"
        color = "#ff4444"
        tips = "Consult a doctor immediately. Avoid sugar and monitor glucose daily."

    details = (
        f"Your total score is {score}. Based on age, BMI and glucose value, "
        f"your calculated risk category is {risk}."
    )

    return jsonify({
        "prediction": risk,
        "color": color,
        "details": details,
        "tips": tips,
        "probability": None
    })


# ---------------------------
# GRAPH API
# ---------------------------
@app.route("/graph-data")
def graph_data():
    return jsonify({
        "glucose": df["Glucose"].tolist(),
        "bmi": df["BMI"].tolist(),
        "age": df["Age"].tolist(),
        "outcome": df["Outcome"].tolist(),
    })


# ---------------------------
# AUTO OPEN BROWSER
# ---------------------------
def open_browser():
    webbrowser.open("http://127.0.0.1:5000")


if __name__ == "__main__":
    threading.Timer(1, open_browser).start()
    app.run(debug=True)