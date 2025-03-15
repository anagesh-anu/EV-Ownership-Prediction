####################################################################################
# 📜 Install Streamlit
# pip install streamlit pandas joblib scikit-learn

#  How to Run the Streamlit Web App
# streamlit run ev_app.py

# This will open a Web UI with sliders and buttons to interactively predict EV ownership.

####################################################################################
import streamlit as st
import joblib
import pandas as pd

# Load model and scaler
#xgb_load = joblib.load(r"notebooks\xgb_model.pkl")
#cat_loaded = joblib.load(r"notebooks\cat_encoders.pkl")
#loaded_scalers = joblib.load(r"notebooks\scalers_dict.pkl")


xgb_load = joblib.load("xgb_model.pkl")
cat_loaded = joblib.load("cat_encoders.pkl")
loaded_scalers = joblib.load("scalers_dict.pkl")



# Web UI Title
st.title("🚗 EV Ownership Prediction")

# User Inputs
Zip_Code = st.selectbox('Select the Zip Code from below',('20001', '20702', '20005', '20002', '20801', '20004', '20701', '20802', '20602', '20006', '20601','20003'))
Government_Subsidy = st.selectbox('Does the Governmet Has subsidy in that City',('YES','NO'))
Home_Type = st.selectbox("Select your home type", ('Apartment','Townhouse','Detached House'))
Num_Residents = int(st.selectbox("Select your number of residents", ('1','2','3','4','5','6','7','8','9','10')))
Nearby_Charging_Station_Distance_miles = st.number_input("Enter the near by charging station distance in miles", min_value=0.0)
Household_Income_kUSD = st.number_input("Approximate Household income", min_value=0.0)
Has_Solar_Panels = int(st.selectbox("Has Solar panels installed", ('1','0')))
Geographic_Location_Score = st.number_input("Geographic Location Score", min_value=0.0)
Power_Factor_Variation = st.number_input("Power Factor Variation of that home", min_value=0.0)
Avg_Temperature_C = st.number_input("Avg Temperature Observed", min_value=0.0)
Sunlight_Hours_Per_Day = st.number_input("Average Sunlight Hours Per_Day", min_value=0.0)
Rainfall_mm = st.number_input("Average Rainfall Observed(in mm)", min_value=0.0)
Snowfall_cm = st.number_input("Average Snowfall Observed (in cm)", min_value=0.0)
Total_night_kWh = st.number_input("Average Power Consumption in Night hours", min_value=0.0)


# Predict Button
if st.button("Predict EV Ownership"):
    # Create input DataFrame
    test_df = pd.DataFrame([[Zip_Code, Government_Subsidy, Home_Type, Num_Residents, Nearby_Charging_Station_Distance_miles, 
                                Household_Income_kUSD,Has_Solar_Panels, Geographic_Location_Score,Power_Factor_Variation,
                                Avg_Temperature_C, Sunlight_Hours_Per_Day,Rainfall_mm, Snowfall_cm,Total_night_kWh]],
                              columns=['Zip_Code', 'Government_Subsidy', 'Home_Type', 'Num_Residents',
                                       'Nearby_Charging_Station_Distance_miles', 'Household_Income_kUSD',
                                       'Has_Solar_Panels', 'Geographic_Location_Score',
                                       'Power_Factor_Variation', 'Avg_Temperature_C', 'Sunlight_Hours_Per_Day',
                                       'Rainfall_mm', 'Snowfall_cm', 'Total_night_kWh'])
    
    # Scale numerical features
    # Apply transformations to numerical  data
    for col, scaler in loaded_scalers.items():
        test_df[col] = scaler.transform(test_df[[col]])

    
    # Apply categorical transformations to test data
    for col, cat in cat_loaded.items():
        test_df[col] = cat.transform(test_df[[col]]) 

    # Prediction
    prediction = xgb_load.predict(test_df)
    result = "✅ EV Present" if prediction[0] == 1 else "❌ No EV"

    # Show result
    st.success(f"Prediction: {result}")
