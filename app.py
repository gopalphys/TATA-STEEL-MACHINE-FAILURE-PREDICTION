import streamlit as st
import numpy as np
import pandas as pd
import pickle
import joblib




# Define the numeric columns that will be input by the user
columns_name = ['Air_temperature_K',
 'Rotational_speed_rpm',
 'Torque_Nm',
 'Tool_wear_min',
 'TWF',
 'HDF',
 'PWF',
 'OSF',
 'RNF']
 
# Title for the Streamlit app
st.title("⚙️🛠️ Tata Steel Machine Failure Prediction")
st.write("#### This is the Tata Steel Machine Failure App. If you provide the required input data, the app will predict whether there is a risk of machine failure or not. ")

# Numeric inputs: Age, RestingBP, Cholesterol, Fasting Blood Sugar, MaxHR, and Oldpeak
Air_temperature = st.number_input("🔥 **Air_temperature_K** (min_value=295, max_value=305)", min_value=295, max_value=305)
Rotational_speed = st.number_input("🔄**Rotational_speed_rpm** (min_value=1150, max_value=2900)", min_value=1150, max_value=2900 )
Torque_Nm = st.number_input(" 💪🏼**Torque_Nm** (min_value=3.5, max_value=80.0) ", min_value=3.5, max_value=80.0)
Tool_wear_min = st.number_input("🛠️ **Tool_wear_min** (min_value=0, max_value=260)", min_value=0, max_value=260)
TWF = st.selectbox("**TWF** [0,1]", [0,1])
HDF = st.selectbox("**HDF** [0,1]", [0,1])
PWF = st.selectbox("**PWF** [0,1]", [0,1])
OSF = st.selectbox("**OSF** [0,1]", [0,1])
RNF = st.selectbox("**RNF** [0,1]", [0,1])



# When the user clicks the "Predict" button, the prediction process is triggered
if st.button("Predict"):
    # Prepare numeric features from user input
  features = [Air_temperature, Rotational_speed, Torque_Nm, Tool_wear_min, 
   TWF,HDF,PWF,OSF,RNF]

  input_df = pd.DataFrame([features], columns=columns_name)
  st.write("### The given input parameter values")
  st.write(input_df,
  "#### Predicting... ")
  

  selector=joblib.load('selector.pkl')
  X_transform=selector.transform(input_df)
  


    # Load the trained model from a local file
  model = joblib.load("models_over_sampling/best_model.pkl")

    # Predict the risk of heart disease based on the input features
  prediction = model.predict(X_transform)

    # Display the prediction result to the user
  if prediction == 1:
   st.error("⚠️  High risk of Machine Failure!")  # Display error if high risk
  else:
    st.success("✅  Low risk of Machine Failure!")  # Display success if low risk


