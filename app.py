import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle
from tensorflow.keras.models import load_model

# Load Model
try:
    model = load_model("model.h5")
except Exception as e:
    st.error(f"Error loading model: {e}")

# Load Encoders and Scaler
try:
    with open("label_Onehot_geograpy.pkl", "rb") as file:
        label_encoder_geo = pickle.load(file)

    with open("label_encoder_gender.pkl", "rb") as file:
        label_encoder_gender = pickle.load(file)

    with open("scaler.pkl", "rb") as file:
        scaler = pickle.load(file)
except Exception as e:
    st.error(f"Error loading encoders or scaler: {e}")

# Streamlit UI
st.title("Customer Churn Prediction")

geography = st.selectbox('Geography', label_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 90)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Prepare Input Data
try:
    input_data = pd.DataFrame({
        "CreditScore": [credit_score],
        "Gender": [label_encoder_gender.transform([gender])[0]],
        "Age": [age],
        "Tenure": [tenure],
        "Balance": [balance],
        "NumOfProducts": [num_of_products],
        "HasCrCard": [has_cr_card],
        "IsActiveMember": [is_active_member],
        "EstimatedSalary": [estimated_salary],
    })

    ge_encoded = label_encoder_geo.transform([[geography]]).toarray()
    ge_encoded_df = pd.DataFrame(ge_encoded, columns=label_encoder_geo.get_feature_names_out(["Geography"]))

    input_data = pd.concat([input_data.reset_index(drop=True), ge_encoded_df], axis=1)

    input_data_scaled = scaler.transform(input_data)

    prediction = model.predict(input_data_scaled)
    prediction_prob = prediction[0][0]

    st.write(f'Churn Probability: {prediction_prob:.2f}')
    st.write("The customer is likely to churn" if prediction_prob > 0.5 else "The customer is not likely to churn")

except Exception as e:
    st.error(f"Error processing input data: {e}")
  