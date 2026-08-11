import streamlit as st
import pandas as pd
import joblib
import json
from tensorflow import keras

# ==========================================
# LOAD ARTIFACTS
# ==========================================

scaler = joblib.load("artifacts/scaler.joblib")

model = keras.models.load_model(
    "artifacts/churn_model.keras"
)

with open("artifacts/metadata.json", "r") as f:
    metadata = json.load(f)

threshold = metadata["threshold"]
feature_columns = metadata["feature_columns"]


# ==========================================
# STREAMLIT APP
# ==========================================

st.title("Customer Churn Prediction")

st.write("Enter customer details below")


# ==========================================
# INPUTS
# ==========================================

credit_score = st.number_input(
    "Credit Score",
    min_value=300,
    max_value=850,
    value=650
)

age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=35
)

tenure = st.number_input(
    "Tenure",
    min_value=0,
    max_value=20,
    value=5
)

balance = st.number_input(
    "Balance",
    min_value=0.0,
    value=50000.0
)

num_of_products = st.number_input(
    "Number of Products",
    min_value=1,
    max_value=4,
    value=1
)

has_cr_card = st.selectbox(
    "Has Credit Card",
    [0, 1]
)

is_active_member = st.selectbox(
    "Is Active Member",
    [0, 1]
)

estimated_salary = st.number_input(
    "Estimated Salary",
    min_value=0.0,
    value=50000.0
)

geography = st.selectbox(
    "Geography",
    ["France", "Germany", "Spain"]
)

gender = st.selectbox(
    "Gender",
    ["Female", "Male"]
)


# ==========================================
# PREDICTION
# ==========================================

if st.button("Predict Churn"):

    # Create customer DataFrame
    new_customer = pd.DataFrame({
        "credit_score": [credit_score],
        "age": [age],
        "tenure": [tenure],
        "balance": [balance],
        "num_of_products": [num_of_products],
        "has_cr_card": [has_cr_card],
        "is_active_member": [is_active_member],
        "estimated_salary": [estimated_salary],
        "geography": [geography],
        "gender": [gender]
    })

    # One-hot encoding
    X_new = pd.get_dummies(
        new_customer,
        columns=["geography", "gender"],
        drop_first=True
    )

    # Match training columns
    X_new = X_new.reindex(
        columns=feature_columns,
        fill_value=0
    )

    # Scaling
    X_new_scaled = scaler.transform(X_new)

    # Prediction probability
    probability = model.predict(
        X_new_scaled,
        verbose=0
    ).ravel()[0]

    # Apply saved threshold
    prediction = int(
        probability >= threshold
    )


    # ======================================
    # RESULT
    # ======================================

    st.subheader("Prediction Result")

    st.write(
        f"Churn Probability: **{probability:.2%}**"
    )

    st.write(
        f"Threshold: **{threshold:.2f}**"
    )

    if prediction == 1:
        st.error("⚠️ Customer is likely to CHURN")
    else:
        st.success("✅ Customer is likely to STAY")