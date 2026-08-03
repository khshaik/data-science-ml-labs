"""
Streamlit Web Dashboard for Churn Prediction
Interactive interface for predictions and monitoring

Bonus Feature: Web Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json

# Page configuration
st.set_page_config(
    page_title="Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# API configuration
API_URL = "http://localhost:8000"

# Title
st.markdown('<h1 class="main-header">🎯 Customer Churn Prediction Dashboard</h1>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    api_url = st.text_input("API URL", value=API_URL)
    
    st.markdown("---")
    st.header("📊 Navigation")
    page = st.radio("Select Page", ["Single Prediction", "Batch Prediction", "Model Monitoring"])
    
    st.markdown("---")
    st.info("💡 **Tip**: Use this dashboard to predict customer churn and monitor model performance.")

# Check API health
def check_api_health():
    try:
        response = requests.get(f"{api_url}/health", timeout=5)
        if response.status_code == 200:
            return True, response.json()
        return False, None
    except:
        return False, None

# Display API status
api_healthy, health_info = check_api_health()
if api_healthy:
    st.sidebar.success(f"✅ API Connected")
    if health_info:
        st.sidebar.text(f"Model: {health_info.get('model_version', 'Unknown')}")
else:
    st.sidebar.error("❌ API Disconnected")
    st.error("⚠️ Cannot connect to API. Please start the API server: `uvicorn src.serving.api:app --port 8000`")

# Page 1: Single Prediction
if page == "Single Prediction":
    st.header("🔮 Single Customer Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Customer Demographics")
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        
        st.subheader("Account Information")
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0, 0.5)
        total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 780.0, 10.0)
    
    with col2:
        st.subheader("Services")
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
        
        st.subheader("Contract Details")
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox("Payment Method", 
                                     ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    
    if st.button("🎯 Predict Churn", type="primary", use_container_width=True):
        if not api_healthy:
            st.error("Cannot make prediction: API is not available")
        else:
            # Prepare request
            customer_data = {
                "gender": gender,
                "SeniorCitizen": senior_citizen,
                "Partner": partner,
                "Dependents": dependents,
                "tenure": tenure,
                "PhoneService": phone_service,
                "MultipleLines": multiple_lines,
                "InternetService": internet_service,
                "OnlineSecurity": online_security,
                "OnlineBackup": online_backup,
                "DeviceProtection": device_protection,
                "TechSupport": tech_support,
                "StreamingTV": streaming_tv,
                "StreamingMovies": streaming_movies,
                "Contract": contract,
                "PaperlessBilling": paperless_billing,
                "PaymentMethod": payment_method,
                "MonthlyCharges": monthly_charges,
                "TotalCharges": total_charges
            }
            
            # Make prediction
            with st.spinner("Making prediction..."):
                try:
                    response = requests.post(f"{api_url}/predict", json=customer_data, timeout=10)
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Display results
                        st.success("✅ Prediction Complete!")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Churn Prediction", result['churn_prediction'])
                        with col2:
                            st.metric("Churn Probability", f"{result['churn_probability']:.1%}")
                        with col3:
                            risk_color = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}
                            st.metric("Risk Level", f"{risk_color.get(result['risk_level'], '')} {result['risk_level']}")
                        with col4:
                            st.metric("Latency", f"{result['latency_ms']:.1f} ms")
                        
                        # Visualization
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number+delta",
                            value=result['churn_probability'] * 100,
                            title={'text': "Churn Probability (%)"},
                            delta={'reference': 50},
                            gauge={
                                'axis': {'range': [None, 100]},
                                'bar': {'color': "darkblue"},
                                'steps': [
                                    {'range': [0, 40], 'color': "lightgreen"},
                                    {'range': [40, 70], 'color': "yellow"},
                                    {'range': [70, 100], 'color': "red"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 70
                                }
                            }
                        ))
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Recommendations
                        if result['churn_prediction'] == "Yes":
                            st.warning("⚠️ **High Churn Risk - Recommended Actions:**")
                            st.markdown("""
                            - 🎁 Offer retention discount or upgrade
                            - 📞 Schedule proactive customer service call
                            - 💳 Review payment method and billing issues
                            - 📱 Promote additional services for engagement
                            """)
                        else:
                            st.success("✅ **Low Churn Risk - Recommended Actions:**")
                            st.markdown("""
                            - 🌟 Continue excellent service
                            - 📊 Monitor satisfaction metrics
                            - 🎯 Consider upsell opportunities
                            """)
                    else:
                        st.error(f"Prediction failed: {response.text}")
                
                except Exception as e:
                    st.error(f"Error: {str(e)}")

# Page 2: Batch Prediction
elif page == "Batch Prediction":
    st.header("📦 Batch Prediction")
    
    st.info("Upload a CSV file with customer data to get churn predictions for multiple customers.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        st.subheader("📊 Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        st.metric("Total Customers", len(df))
        
        if st.button("🚀 Run Batch Prediction", type="primary"):
            st.info("For batch predictions, use the command line tool: `python src/serving/batch_predict.py`")
            st.code(f"python src/serving/batch_predict.py --input {uploaded_file.name} --output predictions.csv", language="bash")

# Page 3: Model Monitoring
elif page == "Model Monitoring":
    st.header("📈 Model Monitoring Dashboard")
    
    col1, col2, col3 = st.columns(3)
    
    # Mock metrics (in production, fetch from Prometheus)
    with col1:
        st.metric("Total Predictions", "12,543", "+234")
    with col2:
        st.metric("Avg Latency", "52 ms", "-3 ms")
    with col3:
        st.metric("Error Rate", "0.02%", "-0.01%")
    
    st.markdown("---")
    
    # Mock performance chart
    st.subheader("📊 Prediction Throughput")
    dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
    throughput = np.random.randint(300, 500, 30)
    
    fig = px.line(x=dates, y=throughput, labels={'x': 'Date', 'y': 'Predictions/day'})
    fig.update_layout(title="Daily Prediction Volume")
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("⏱️ Latency Distribution")
        latencies = np.random.gamma(2, 25, 1000)
        fig = px.histogram(latencies, nbins=50, labels={'value': 'Latency (ms)'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Prediction Distribution")
        predictions = pd.DataFrame({
            'Prediction': ['No Churn', 'Churn'],
            'Count': [7500, 3200]
        })
        fig = px.pie(predictions, values='Count', names='Prediction', 
                    color_discrete_sequence=['#00CC96', '#EF553B'])
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.info("💡 For real-time monitoring, access Grafana dashboard at http://localhost:3000")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Enterprise MLOps Churn Prediction System | Built with Streamlit</p>
    <p>Model Version: v1.0.0 | Last Updated: 2024</p>
</div>
""", unsafe_allow_html=True)
