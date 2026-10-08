import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="House Price Prediction", page_icon="🏠")
st.title("🏠 House Price Prediction")
st.write("Predict a house price using a trained machine learning model.")

model_path = Path("models/house_price_model.pkl")

if not model_path.exists():
    st.warning("Model not trained yet. Run: py src/train.py")
else:
    model = joblib.load(model_path)
    st.subheader("Enter house details")
    area = st.number_input("Area (sq ft)", min_value=100.0, value=1000.0)
    bedrooms = st.number_input("Bedrooms", min_value=1, value=2, step=1)
    bathrooms = st.number_input("Bathrooms", min_value=1, value=1, step=1)
    age = st.number_input("Age of house (years)", min_value=0, value=5, step=1)
    if st.button("Predict Price"):
        X = pd.DataFrame([{"area": area, "bedrooms": bedrooms, "bathrooms": bathrooms, "age": age}])
        prediction = model.predict(X)[0]
        st.success(f"Predicted Price: ₹{prediction:,.0f}")
