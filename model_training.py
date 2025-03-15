# 🔹 What This Script Does

# ✅ Loads the preprocessed dataset
# ✅ Splits data into training (80%) & testing (20%)
# ✅ Scales features (StandardScaler for better model performance)
# ✅ Trains a Random Forest Classifier
# ✅ Predicts and evaluates accuracy, precision, recall, and F1-score
# ✅ Displays feature importance (Which features impact Has_EV most)
# ✅ Saves the trained model (ev_prediction_model.pkl)



import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Load the processed dataset
df = pd.read_csv("ev_prediction_dataset_final.csv")

# Drop Timestamp column (not useful for prediction)
df.drop(columns=['Timestamp'], inplace=True, errors='ignore')

# Define features (X) and target (y)
X = df.drop(columns=['Has_EV'])  # Features
y = df['Has_EV']  # Target

# Train-Test Split (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale the dataset (Only needed if using distance-based models like KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=10)
rf_model.fit(X_train_scaled, y_train)

# Predictions
y_pred = rf_model.predict(X_test_scaled)

# Evaluate Model
accuracy = accuracy_score(y_test, y_pred)
print(f"🔹 Model Accuracy: {accuracy:.4f}")

# Detailed Classification Report
print("\n📌 Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
print("\n🔍 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Feature Importance Analysis
feature_importance = pd.Series(rf_model.feature_importances_, index=X.columns)
print("\n📊 Feature Importance Ranking:")
print(feature_importance.sort_values(ascending=False))

# Save trained model
import joblib
joblib.dump(rf_model, "ev_prediction_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("\n✅ Model saved as 'ev_prediction_model.pkl' & Scaler saved as 'scaler.pkl'")

########################################################
## Hyper parameter tuning and saving the best model ####
########################################################

# Define the Random Forest model
rf_model = RandomForestClassifier(random_state=42)

# Define hyperparameter grid
param_grid = {
    'n_estimators': [100, 200, 300],  # Number of trees
    'max_depth': [10, 15, 20, None],  # Tree depth
    'min_samples_split': [2, 5, 10],  # Minimum samples per split
    'min_samples_leaf': [1, 2, 4],  # Minimum samples per leaf
    'max_features': ['sqrt', 'log2']  # Feature selection per tree
}

# Perform Grid Search with 5-fold Cross-Validation
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
grid_search.fit(X_train_scaled, y_train)

# Get the best parameters
best_params = grid_search.best_params_
print(f"\n✅ Best Hyperparameters: {best_params}")

# Train the best model
best_rf_model = RandomForestClassifier(**best_params, random_state=42)
best_rf_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = best_rf_model.predict(X_test_scaled)

# Evaluate the tuned model
accuracy = accuracy_score(y_test, y_pred)
print(f"\n🔹 Tuned Model Accuracy: {accuracy:.4f}")

# Detailed Classification Report
print("\n📌 Classification Report:")
print(classification_report(y_test, y_pred))

# Save the best model
import joblib
joblib.dump(best_rf_model, "ev_prediction_best_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("\n✅ Best tuned model saved as 'ev_prediction_best_model.pkl'")

##########################################################################
# Would you like to:
# 🔹 Compare this model with XGBoost or Logistic Regression?
# 🔹 Deploy the model in a real-world application (Flask, FastAPI, etc.)?
# 🔹 Visualize the most important features in predicting Has_EV?
#########################################################################

