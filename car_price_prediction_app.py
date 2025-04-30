import streamlit as st
import pandas as pd
import numpy as np
import joblib  # To load the model and scaler

# Load the trained Random Forest model and scaler
rf_model = joblib.load('rf_model.pkl')  # Load the trained Random Forest model
scaler = joblib.load('scaler.pkl')  # Load the scaler

# Define Streamlit app structure
def predict_price(inputs):
    # Create a DataFrame with the user input
    input_df = pd.DataFrame([inputs], columns=["Wheelbase(mm)", "Length(mm)", "Performance of EM(hp)", "Battery Capacity(KWh)", "Battery Size(L)", "Passenger / Seating Capacity"])

    # Scale the input data using the loaded scaler
    input_scaled = scaler.transform(input_df)

    # Predict the car price using the trained model
    price_prediction = rf_model.predict(input_scaled)
    return price_prediction[0]

# Title for the web app
st.title("Car Price Prediction App")

# Get user input using Streamlit's text input and number input widgets
marque_origin = st.selectbox("Select Marque Origin", ["Europe", "China", "US", "Other"])
year_production = st.number_input("Enter Calendar Year of Production", min_value=2016, max_value=2025, step=1, value=2023)
n_doors = st.number_input("Number of Doors", min_value=2, max_value=5, value=4)
body_style = st.selectbox("Select Body Style", ["SUV", "Sedan", "Hatchback", "Coupe", "Convertible"])
drivetrain = st.selectbox("Select Drivetrain", ["AWD", "RWD", "FWD"])
wheelbase = st.number_input("Wheelbase (mm)", min_value=2000, max_value=5000, value=2800)
length = st.number_input("Length (mm)", min_value=2500, max_value=6000, value=4500)
performance_hp = st.number_input("Performance of Electric Motor (hp)", min_value=50, max_value=1500, value=250)
battery_capacity = st.number_input("Battery Capacity (KWh)", min_value=10, max_value=200, value=75)
battery_size = st.number_input("Battery Size (L)", min_value=50, max_value=800, value=400)
passenger_capacity = st.number_input("Passenger / Seating Capacity", min_value=2, max_value=9, value=5)

# User input as dictionary
user_input = {
    "Wheelbase(mm)": wheelbase,
    "Length(mm)": length,
    "Performance of EM(hp)": performance_hp,
    "Battery Capacity(KWh)": battery_capacity,
    "Battery Size(L)": battery_size,
    "Passenger / Seating Capacity": passenger_capacity
}

# Button to get prediction
if st.button("Predict Car Price"):
    # Call the prediction function
    predicted_price = predict_price(user_input)
    
    # Display the result
    st.write(f"💰 The predicted car price is: {predicted_price:.2f} Euros")
