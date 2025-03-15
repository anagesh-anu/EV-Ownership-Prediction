######################################################################################################
# 🔹 How to Run the Flask Web App

# 1️⃣ Install Flask (if not already installed)
# pip install flask pandas joblib scikit-learn

# 2️⃣ Run the Flask app
# python ev_api.py

# 3️⃣ Open in Browser
# Go to: 👉 http://127.0.0.1:5000/

# You will see a simple web form where you can input values and get the EV Prediction result.

######################################################################################################

import joblib
import pandas as pd
from flask import Flask, request, jsonify, render_template

# Load trained model and scaler
model = joblib.load("ev_prediction_best_model.pkl")
scaler = joblib.load("scaler.pkl")

# Initialize Flask app
app = Flask(__name__)

# Home Page with Web Form
@app.route('/')
def home():
    return render_template("index.html")  # Load HTML page

# Prediction API (For Web & API Calls)
@app.route('/predict', methods=['POST'])
def predict_ev():
    try:
        # Get form data
        data = request.form  
        
        # Convert input into a DataFrame
        input_data = pd.DataFrame([data])

        # Convert numeric values
        numeric_features = [
            'Total_Electricity_Usage_kWh', 'Household_Income_kUSD', 
            'Nearby_Charging_Station_Distance_km', 'Daily_Commute_Distance_km', 
            'Electricity_Cost_Per_kWh', 'Cost_of_Living_Index'
        ]
        input_data[numeric_features] = input_data[numeric_features].astype(float)

        # Scale numerical features
        input_data[numeric_features] = scaler.transform(input_data[numeric_features])
        
        # Predict using the model
        prediction = model.predict(input_data)
        result = "✅ EV Present" if prediction[0] == 1 else "❌ No EV"

        return render_template("index.html", prediction_text=result)

    except Exception as e:
        return jsonify({"error": str(e)})

# Run Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
