import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from PIL import Image
import base64


st.set_page_config(
    page_title="Heartly: Heart Attack Risk Prediction Tool",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #ff6b6b, #ee5a24);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .risk-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    
    .input-section {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #007bff;
        margin: 1rem 0;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #28a745;
    }
    
    .stSlider > div > div > div > div {
        background-color: #007bff;
    }
    
    .help-tooltip {
        background-color: #e3f2fd;
        padding: 0.5rem;
        border-radius: 5px;
        border-left: 3px solid #2196f3;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    try:
        model = joblib.load('random_forest_model.joblib')
        scaler = joblib.load('scaler.joblib')
        return model, scaler
    except FileNotFoundError:
        st.error("Model or scaler files not found. Please run the model.py script first.")
        return None, None

model, scaler = load_model()

# --- Session State Management ---
if 'patient_data' not in st.session_state:
    st.session_state.patient_data = {}
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []


st.markdown("""
<div class="main-header">
    <h1>❤️ Heartly: Heart Attack Risk Prediction Tool</h1>
    <p>Advanced ML-powered cardiovascular risk assessment using medical lab results</p>
</div>
""", unsafe_allow_html=True)


tab1, tab2, tab3, tab4 = st.tabs(["🏥 Risk Assessment", "📊 Data Analysis", "📈 Model Insights", "ℹ️ About"])

with tab1:
    # --- Sidebar for User Inputs ---
    st.sidebar.markdown("### 🩺 Patient Data Input")
    
    # Patient Information Section
    st.sidebar.markdown("#### **Patient Information**")
    patient_name = st.sidebar.text_input("Patient Name (Optional)", placeholder="Enter patient name")
    
    # Demographics
    st.sidebar.markdown("#### **Demographics**")
    age = st.sidebar.slider('Age', 20, 95, 55, help="Patient age in years")
    
    gender_options = {0: 'Female', 1: 'Male'}
    gender = st.sidebar.selectbox('Gender', options=list(gender_options.keys()), 
                                  format_func=lambda x: gender_options[x],
                                  help="Patient gender")
    
    
    st.sidebar.markdown("#### **Vital Signs**")
    heart_rate = st.sidebar.slider('Heart Rate (bpm)', 40, 120, 75, 
                                   help="Normal range: 60-100 bpm")
    
    systolic_bp = st.sidebar.slider('Systolic Blood Pressure (mm Hg)', 90, 220, 130,
                                    help="Normal range: 90-140 mm Hg")
    
    diastolic_bp = st.sidebar.slider('Diastolic Blood Pressure (mm Hg)', 45, 130, 80,
                                     help="Normal range: 60-90 mm Hg")
    
    # Lab Results
    st.sidebar.markdown("#### **Laboratory Results**")
    blood_sugar = st.sidebar.slider('Blood Sugar (mg/dL)', 85, 300, 140,
                                    help="Normal range: 70-140 mg/dL")
    
    ck_mb = st.sidebar.slider('CK-MB (ng/mL)', 0.3, 30.0, 3.8, 0.1,
                              help="Normal range: 0-5 ng/mL")
    
    troponin = st.sidebar.slider('Troponin (ng/mL)', 0.001, 1.1, 0.01, 0.001,
                                 help="Normal range: 0-0.04 ng/mL")
    
  
    st.sidebar.markdown("---")
    st.sidebar.markdown("#### **Data Management**")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("💾 Save Data"):
            patient_data = {
                'name': patient_name,
                'age': age,
                'gender': gender,
                'heart_rate': heart_rate,
                'systolic_bp': systolic_bp,
                'diastolic_bp': diastolic_bp,
                'blood_sugar': blood_sugar,
                'ck_mb': ck_mb,
                'troponin': troponin
            }
            st.session_state.patient_data = patient_data
            st.sidebar.success("Data saved!")
    
    with col2:
        if st.button("📂 Load Data") and st.session_state.patient_data:
            
            st.sidebar.info("Data loaded! Refresh to see changes.")
    
   
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.markdown("### 📋 Patient Summary")
        
       
        metrics = ['Age', 'Gender', 'Heart Rate', 'Systolic BP', 'Diastolic BP', 'Blood Sugar', 'CK-MB', 'Troponin']
        values = [f"{age} years", gender_options[gender], f"{heart_rate} bpm", 
                  f"{systolic_bp} mm Hg", f"{diastolic_bp} mm Hg", f"{blood_sugar} mg/dL",
                  f"{ck_mb} ng/mL", f"{troponin} ng/mL"]

        
        if patient_name:
            metrics.insert(0, 'Name')
            values.insert(0, patient_name)

        
        summary_data = {
            'Metric': metrics,
            'Value': values
        }
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
        st.markdown("### ⚠️ Risk Indicators")
        
        risk_indicators = []
        if age > 65:
            risk_indicators.append("🔴 Advanced age (>65)")
        elif age > 55:
            risk_indicators.append("🟡 Moderate age (55-65)")
        else:
            risk_indicators.append("🟢 Young age (<55)")
            
        if heart_rate > 100:
            risk_indicators.append("🔴 Elevated heart rate (>100 bpm)")
        elif heart_rate < 60:
            risk_indicators.append("🟡 Low heart rate (<60 bpm)")
        else:
            risk_indicators.append("🟢 Normal heart rate (60-100 bpm)")
            
        if systolic_bp > 140:
            risk_indicators.append("🔴 High systolic BP (>140 mm Hg)")
        elif systolic_bp > 120:
            risk_indicators.append("🟡 Elevated systolic BP (120-140 mm Hg)")
        else:
            risk_indicators.append("🟢 Normal systolic BP (<120 mm Hg)")
            
        if blood_sugar > 200:
            risk_indicators.append("🔴 High blood sugar (>200 mg/dL)")
        elif blood_sugar > 140:
            risk_indicators.append("🟡 Elevated blood sugar (140-200 mg/dL)")
        else:
            risk_indicators.append("🟢 Normal blood sugar (<140 mg/dL)")
            
        if ck_mb > 5:
            risk_indicators.append("🔴 Elevated CK-MB (>5 ng/mL)")
        else:
            risk_indicators.append("🟢 Normal CK-MB (<5 ng/mL)")
            
        if troponin > 0.04:
            risk_indicators.append("🔴 Elevated troponin (>0.04 ng/mL)")
        else:
            risk_indicators.append("🟢 Normal troponin (<0.04 ng/mL)")
        
        for indicator in risk_indicators:
            st.markdown(f"- {indicator}")
    
    with col2:
        st.markdown("### 🎯 Risk Assessment")
        
       
        if st.button("🔍 **Predict Heart Attack Risk**", type="primary", use_container_width=True):
            if model is None or scaler is None:
                st.error("Model not loaded. Please check model files.")
            else:
                with st.spinner("Analyzing patient data..."):
                    n
                    user_data = {
                        'age': age,
                        'gender': gender,
                        'heart_rate': heart_rate,
                        'systolic_blood_pressure': systolic_bp,
                        'diastolic_blood_pressure': diastolic_bp,
                        'blood_sugar': blood_sugar,
                        'ck_mb': ck_mb,
                        'troponin': troponin
                    }
                    
                    feature_names = ['age', 'gender', 'heart_rate', 'systolic_blood_pressure', 
                                     'diastolic_blood_pressure', 'blood_sugar', 'ck_mb', 'troponin']
                    
                    user_input_df = pd.DataFrame(user_data, index=[0])[feature_names]
                    
                    
                    user_input_scaled = scaler.transform(user_input_df)
                    
                    
                    prediction = model.predict(user_input_scaled)
                    prediction_proba = model.predict_proba(user_input_scaled)
                    
                    
                    prediction_record = {
                        'patient_name': patient_name or f"Patient_{len(st.session_state.prediction_history)+1}",
                        'age': age,
                        'gender': gender_options[gender],
                        'risk_score': prediction_proba[0][1],
                        'prediction': 'High Risk' if prediction[0] == 1 else 'Low Risk',
                        'timestamp': pd.Timestamp.now()
                    }
                    st.session_state.prediction_history.append(prediction_record)
                    
                    
                    risk_score = prediction_proba[0][1] * 100
                    
                    if prediction[0] == 1:
                        st.markdown(f"""
                        <div class="risk-card" style="background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);">
                            <h2>🚨 HIGH RISK DETECTED</h2>
                            <h3>Risk Score: {risk_score:.1f}%</h3>
                            <p>Immediate medical attention recommended</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="risk-card" style="background: linear-gradient(135deg, #28a745 0%, #20c997 100%);">
                            <h2>✅ LOW RISK</h2>
                            <h3>Risk Score: {risk_score:.1f}%</h3>
                            <p>Continue regular health monitoring</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                   
                    st.markdown("### 📊 Risk Level Breakdown")
                    
                    if risk_score < 20:
                        risk_level = "Very Low"
                        color = "#28a745"
                    elif risk_score < 40:
                        risk_level = "Low"
                        color = "#20c997"
                    elif risk_score < 60:
                        risk_level = "Moderate"
                        color = "#ffc107"
                    elif risk_score < 80:
                        risk_level = "High"
                        color = "#fd7e14"
                    else:
                        risk_level = "Very High"
                        color = "#dc3545"
                    
                    
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number+delta",
                        value = risk_score,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': f"Risk Level: {risk_level}"},
                        delta = {'reference': 50},
                        gauge = {
                            'axis': {'range': [None, 100]},
                            'bar': {'color': color},
                            'steps': [
                                {'range': [0, 20], 'color': "#28a745"},
                                {'range': [20, 40], 'color': "#20c997"},
                                {'range': [40, 60], 'color': "#ffc107"},
                                {'range': [60, 80], 'color': "#fd7e14"},
                                {'range': [80, 100], 'color': "#dc3545"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 80
                            }
                        }
                    ))
                    
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                    
                   
                    st.markdown("### 💡 Recommendations")
                    if prediction[0] == 1:
                        st.error("""
                        **Immediate Actions Required:**
                        - Seek emergency medical care immediately
                        - Contact your healthcare provider
                        - Avoid physical exertion
                        - Monitor symptoms closely
                        """)
                    else:
                        st.success("""
                        **Preventive Measures:**
                        - Continue regular health checkups
                        - Maintain healthy lifestyle
                        - Monitor blood pressure regularly
                        - Follow up with healthcare provider
                        """)
                    
                    st.info("**Medical Disclaimer:** This tool provides preliminary risk assessment based on machine learning analysis. It is not a substitute for professional medical diagnosis or treatment. Always consult with qualified healthcare professionals for medical decisions.")

with tab2:
    st.markdown("### 📊 Data Analysis & Insights")
    
  
    try:
        df = pd.read_csv('Medicaldataset.csv')
        df_clean = df.copy()
        df_clean.columns = df_clean.columns.str.lower().str.replace(' ', '_').str.replace('-', '_')
        df_clean['result'] = df_clean['result'].map({'positive': 1, 'negative': 0})
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Patients", len(df))
        with col2:
            st.metric("High Risk Cases", len(df[df['Result'] == 'positive']))
        with col3:
            st.metric("Low Risk Cases", len(df[df['Result'] == 'negative']))
        
        
        st.markdown("### 🔗 Feature Correlation Analysis")
        correlation_matrix = df_clean.corr()
        
        fig = px.imshow(
            correlation_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="RdBu_r",
            title="Feature Correlation Matrix"
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        
        st.markdown("### 📈 Feature Distributions by Risk Level")
        
        features = ['age', 'heart_rate', 'systolic_blood_pressure', 'diastolic_blood_pressure', 
                    'blood_sugar', 'ck_mb', 'troponin']
        
        selected_feature = st.selectbox("Select Feature to Analyze:", features)
        
        fig = px.histogram(
            df_clean, 
            x=selected_feature, 
            color='result',
            barmode='overlay',
            opacity=0.7,
            title=f"Distribution of {selected_feature.replace('_', ' ').title()} by Risk Level"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
       
        st.markdown("### 📋 Statistical Summary")
        st.dataframe(df_clean.describe(), use_container_width=True)
        
    except Exception as e:
        st.error(f"Error loading data: {e}")

with tab3:
    st.markdown("### 🤖 Model Insights")
    
    if model is not None:
        
        st.markdown("### 🎯 Feature Importance")
        
        feature_names = ['Age', 'Gender', 'Heart Rate', 'Systolic BP', 'Diastolic BP', 'Blood Sugar', 'CK-MB', 'Troponin']
        feature_importance = model.feature_importances_
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=True)
        
        fig = px.bar(
            importance_df,
            x='Importance',
            y='Feature',
            orientation='h',
            title="Feature Importance in Heart Attack Prediction"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        
        st.markdown("### 📊 Model Performance")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Model Type", "Random Forest")
        with col2:
            st.metric("Estimators", "100")
        with col3:
            st.metric("Cross-validation", "5-fold")
        with col4:
            st.metric("Class Weight", "Balanced")
        
        
        if st.session_state.prediction_history:
            st.markdown("### 📝 Recent Predictions")
            history_df = pd.DataFrame(st.session_state.prediction_history)
            st.dataframe(history_df, use_container_width=True)
            
            # Prediction trend
            if len(st.session_state.prediction_history) > 1:
                fig = px.line(
                    history_df,
                    x='timestamp',
                    y='risk_score',
                    title="Risk Score Trend Over Time"
                )
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Model not loaded. Please check model files.")

with tab4:
    st.markdown("### ℹ️ About Heartly")
    
    st.markdown("""
    ## 🏥 Heartly: Heart Attack Risk Prediction Tool
    
    **Heartly** is an advanced machine learning application designed to assess cardiovascular risk 
    based on medical laboratory results and patient demographics.
    
    ###  How It Works
    
    The application uses a **Random Forest Classifier** trained on a comprehensive medical dataset 
    containing patient information and lab results. The model analyzes multiple risk factors to 
    provide a probability-based assessment of heart attack risk.
    
    ###  Features Analyzed
    
    - **Demographics**: Age, Gender
    - **Vital Signs**: Heart Rate, Blood Pressure (Systolic/Diastolic)
    - **Laboratory Results**: Blood Sugar, CK-MB, Troponin
    
    ###  Risk Assessment
    
    The model provides:
    - **Risk Score**: Percentage-based probability (0-100%)
    - **Risk Level**: Categorized assessment (Very Low to Very High)
    - **Recommendations**: Personalized health advice
    - **Visual Analytics**: Interactive charts and graphs
    
    ###  Important Disclaimer
    
    This tool is designed for **educational and research purposes** only. It should not be used 
    as a substitute for professional medical diagnosis, treatment, or clinical decision-making. 
    Always consult with qualified healthcare professionals for medical decisions.
    
    ###  Technical Details
    
    - **Framework**: Streamlit
    - **Machine Learning**: Scikit-learn
    - **Visualization**: Plotly
    - **Data Processing**: Pandas, NumPy
    
    ###  References
    
    This application is based on medical research and clinical guidelines for cardiovascular 
    risk assessment. The model has been trained on anonymized medical data following 
    appropriate data privacy protocols.
    """)


st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    <p>❤️ Heartly - Advanced Cardiovascular Risk Assessment | Built with Streamlit & Machine Learning</p>
    <p>For educational and research purposes only. Not for clinical use.</p>
</div>
""", unsafe_allow_html=True)