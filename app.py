import streamlit as st
import pandas as pd
import joblib
import os

# Page configuration
st.set_page_config(page_title="AtmoGraph: Supply Chain Predictor", page_icon="📈", layout="centered")

st.title("🌐 AtmoGraph: Supply Chain Ripple Effect & Risk Predictor")
st.write("Infotact Solutions - Data Science & Machine Learning Internship Project")
st.markdown("---")

# Model load karne ka function
@st.cache_resource
def load_model():
    model_path = os.path.join("models", "supply_chain_model.pkl")
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

model = load_model()

if model is None:
    st.error("Trained model not found! Please run 'model_training.py' first.")
else:
    st.subheader("📦 Enter Order & Shipment Details for Risk Prediction")

    # User inputs ke liye form fields
    col1, col2 = st.columns(2)

    with col1:
        shipping_days_real = st.number_input("Days for shipping (real)", min_value=0, max_value=30, value=4)
        shipping_days_sched = st.number_input("Days for shipment (scheduled)", min_value=0, max_value=30, value=3)
        benefit_per_order = st.number_input("Benefit per order ($)", value=35.0)
        sales_per_customer = st.number_input("Sales per customer ($)", value=250.0)

    with col2:
        order_item_qty = st.number_input("Order Item Quantity", min_value=1, max_value=20, value=2)
        order_item_total = st.number_input("Order Item Total ($)", value=250.0)
        product_price = st.number_input("Product Price ($)", value=125.0)
        delivery_delay_days = shipping_days_real - shipping_days_sched

    st.write(f"**Calculated Delivery Delay:** {delivery_delay_days} Day(s)")

    # Prediction Button
    if st.button("Predict Late Delivery Risk"):
        # Input DataFrame banana
        input_data = pd.DataFrame([[
            shipping_days_real, shipping_days_sched, benefit_per_order,
            sales_per_customer, order_item_qty, order_item_total,
            product_price, delivery_delay_days
        ]], columns=[
            'Days for shipping (real)', 'Days for shipment (scheduled)', 
            'Benefit per order', 'Sales per customer', 'Order Item Quantity', 
            'Order Item Total', 'Product Price', 'Delivery_Delay_Days'
        ])

        # Prediction run karna
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)

        st.markdown("---")
        st.subheader("🎯 Prediction Results:")
        
        if prediction[0] == 1:
            st.error(f"⚠️ **High Risk of Late Delivery!** (Confidence: {prediction_proba[0][1]*100:.2f}%)")
            st.write("This order is likely to experience supply chain bottlenecks or ripple effect delays.")
        else:
            st.success(f"✅ **Low Risk / On-Time Delivery Expected!** (Confidence: {prediction_proba[0][0]*100:.2f}%)")
            st.write("This order is on track for timely delivery without significant delays.")