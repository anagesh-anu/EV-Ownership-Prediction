import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# Define parameters
num_samples = 1000  # Number of homes (update as needed)
start_date = datetime(2024, 1, 1)
end_date = datetime(2024, 2, 1)  # 1-day period
intervals_per_day = 96  # 15-minute intervals per day

# Generate unique Customer_IDs
customer_ids = np.arange(1, num_samples + 1)

# List of ZIP codes for Maryland and Washington, D.C.
zip_codes = [
    '20601', '20602', '20701', '20702', '20801', '20802',  # Maryland
    '20001', '20002', '20003', '20004', '20005', '20006'   # Washington, D.C.
]

# Assign a random ZIP code to each customer (remains constant per customer)
customer_zip_codes = np.random.choice(zip_codes, num_samples)

# Assign Government Subsidy (YES or NO)
government_subsidy = np.random.choice(["YES", "NO"], num_samples, p=[0.4, 0.6])

# Generate household features
data = {
    "Customer_ID": list(customer_ids),
    "Zip_Code": list(customer_zip_codes),
    "Government_Subsidy": list(government_subsidy),
    "Home_Type": np.random.choice(["Apartment", "Detached House", "Townhouse"], num_samples, p=[0.3, 0.5, 0.2]),
    "Num_Residents": np.random.randint(1, 6, num_samples),
    "Nearby_Charging_Station_Distance_miles": np.random.uniform(0.5, 20, num_samples),
    "Avg_Temperature_C": np.random.uniform(-10, 40, num_samples),
    "Sunlight_Hours_Per_Day": np.random.uniform(2, 12, num_samples),
    "Rainfall_mm": np.random.uniform(0, 50, num_samples),
    "Snowfall_cm": np.random.uniform(0, 30, num_samples),
    "Household_Income_kUSD": np.random.uniform(30, 150, num_samples),
    "Geographic_Location_Score": np.random.uniform(20, 100, num_samples),
    "Has_Solar_Panels": np.random.choice([0, 1], num_samples, p=[0.7, 0.3]),
    "Power_Factor_Variation": np.random.uniform(0.85, 0.99, num_samples),
    "Daily_Commute_Distance_miles": np.random.uniform(5, 100, num_samples),
    "Has_Private_Parking": np.random.choice([0, 1], num_samples, p=[0.3, 0.7]),
    "Cost_of_Living_Index": np.random.uniform(50, 100, num_samples),
    "Electricity_Cost_Per_kWh": np.random.uniform(0.1, 0.4, num_samples)
}

has_ev = []
for i in range(num_samples):
    # Base probability based on government subsidy
    if data["Government_Subsidy"][i] == "YES":
        base_prob = 0.7  # Higher EV probability
    else:
        base_prob = 0.3  # Lower EV probability

    # Adjust probability based on home type
    if data["Home_Type"][i] == "Detached House":
        base_prob += 0.15  # More likely to have EVs
    elif data["Home_Type"][i] == "Apartment":
        base_prob -= 0.15  # Less likely to have EVs

    # Adjust probability based on income (higher income → more EVs)
    if data["Household_Income_kUSD"][i] > 100:
        base_prob += 0.1
    elif data["Household_Income_kUSD"][i] < 50:
        base_prob -= 0.1

    # Solar panels increase the chance of having an EV (cheaper charging)
    if data["Has_Solar_Panels"][i] == 1:
        base_prob += 0.1

    # Nearby charging stations increase EV adoption
    if data["Nearby_Charging_Station_Distance_miles"][i] < 3:
        base_prob += 0.1  # Very close station
    elif data["Nearby_Charging_Station_Distance_miles"][i] > 15:
        base_prob -= 0.1  # Far from stations

    # Higher geographic location score (better infrastructure) increases EV probability
    if data["Geographic_Location_Score"][i] > 80:
        base_prob += 0.1

    # Daily commute distance (longer commutes favor EV adoption)
    if data["Daily_Commute_Distance_miles"][i] > 50:
        base_prob += 0.1  # Long commute → More EVs

    # Private parking increases EV probability (easier charging)
    if data["Has_Private_Parking"][i] == 1:
        base_prob += 0.15

    # Ensure probability stays in range [0,1]
    ev_prob = min(max(base_prob, 0), 1)

    # Assign EV based on final probability
    has_ev.append(np.random.choice([1, 0], p=[ev_prob, 1 - ev_prob]))

data["Has_EV"] = has_ev


# # Assign Has_EV based on Government Subsidy
# has_ev = []
# for i in range(num_samples):
#     if data["Government_Subsidy"][i] == "YES":
#         has_ev.append(np.random.choice([1, 0], p=[0.7, 0.3]))  # Higher EV probability
#     else:
#         has_ev.append(np.random.choice([1, 0], p=[0.3, 0.7]))  # Lower EV probability

# Initialize dependent variables
num_vehicles = []
charging_cycles = []

for i in range(num_samples):
    # Assign Number of Vehicles (higher if has EV)
    num_vehicles_val = np.random.randint(1, 3) if has_ev[i] else np.random.randint(0, 2)
    num_vehicles.append(num_vehicles_val)

    # Assign Charging Cycles (higher if has EV)
    charging_cycle_val = np.random.randint(1, 7) if has_ev[i] else 0
    charging_cycles.append(charging_cycle_val)

# Update dataset with new values
data["Has_EV"] = has_ev
data["Num_Vehicles"] = num_vehicles
data["Charging_Cycles_Per_Week"] = charging_cycles

# Expand to 15-minute intervals
expanded_data = []
for i in range(num_samples):
    for day in pd.date_range(start=start_date, end=end_date, freq="D"):
        for j in range(intervals_per_day):
            timestamp = day + timedelta(minutes=15 * j)

            # Compute per-interval electricity usage (randomized)
            interval_usage = max(np.random.uniform(0.1, 1.5), 0)

            expanded_data.append([
                timestamp, data["Customer_ID"][i], data["Zip_Code"][i], data["Government_Subsidy"][i], 
                data["Home_Type"][i], data["Num_Residents"][i], 
                data["Nearby_Charging_Station_Distance_miles"][i], data["Has_EV"][i], 
                data["Num_Vehicles"][i], data["Charging_Cycles_Per_Week"][i],
                data["Household_Income_kUSD"][i], data["Has_Solar_Panels"][i], 
                data["Geographic_Location_Score"][i], data["Power_Factor_Variation"][i],
                data["Avg_Temperature_C"][i], data["Sunlight_Hours_Per_Day"][i], 
                data["Rainfall_mm"][i], data["Snowfall_cm"][i], interval_usage
            ])

# Convert to DataFrame
columns = ["Timestamp", "Customer_ID", "Zip_Code", "Government_Subsidy", "Home_Type", 
           "Num_Residents", "Nearby_Charging_Station_Distance_miles", "Has_EV", "Num_Vehicles", 
           "Charging_Cycles_Per_Week", "Household_Income_kUSD", "Has_Solar_Panels", 
           "Geographic_Location_Score", "Power_Factor_Variation", "Avg_Temperature_C", 
           "Sunlight_Hours_Per_Day", "Rainfall_mm", "Snowfall_cm", "Interval_Usage_kWh"]

df = pd.DataFrame(expanded_data, columns=columns)

# Save dataset
df.to_csv("ev_prediction_final_with_zipcode7.csv", index=False)

print("✅ Dataset successfully created without Total_Electricity_Usage_kWh and Nighttime_Usage_kWh!")
