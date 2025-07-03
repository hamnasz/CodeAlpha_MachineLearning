import streamlit as st
import pandas as pd
import numpy as np
from utils.data_processor import DataProcessor
from utils.feature_engineer import FeatureEngineer
from utils.model_trainer import ModelTrainer
from utils.model_evaluator import ModelEvaluator
from utils.bias_detector import BiasDetector

# Configure page settings
st.set_page_config(
    page_title="Credit Scoring Model Platform",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables
if 'data' not in st.session_state:
    st.session_state.data = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'engineered_data' not in st.session_state:
    st.session_state.engineered_data = None
if 'trained_models' not in st.session_state:
    st.session_state.trained_models = {}
if 'model_results' not in st.session_state:
    st.session_state.model_results = {}
if 'best_model' not in st.session_state:
    st.session_state.best_model = None

def main():
    st.title("🏦 Credit Scoring Model Platform")
    st.markdown("""
    ### Production-Ready Credit Risk Assessment System
    
    This platform provides a comprehensive solution for building, evaluating, and deploying credit scoring models.
    Navigate through the different sections using the sidebar to complete the ML pipeline.
    """)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    st.sidebar.markdown("Use the pages above to navigate through the ML pipeline:")
    
    # Pipeline status indicators
    st.sidebar.markdown("### Pipeline Status")
    
    # Data Upload Status
    if st.session_state.data is not None:
        st.sidebar.success("✅ Data Uploaded")
    else:
        st.sidebar.error("❌ Data Upload Required")
    
    # Preprocessing Status
    if st.session_state.processed_data is not None:
        st.sidebar.success("✅ Data Preprocessed")
    else:
        st.sidebar.error("❌ Preprocessing Required")
    
    # Feature Engineering Status
    if st.session_state.engineered_data is not None:
        st.sidebar.success("✅ Features Engineered")
    else:
        st.sidebar.error("❌ Feature Engineering Required")
    
    # Model Training Status
    if st.session_state.trained_models:
        st.sidebar.success("✅ Models Trained")
    else:
        st.sidebar.error("❌ Model Training Required")
    
    # Model Evaluation Status
    if st.session_state.model_results:
        st.sidebar.success("✅ Models Evaluated")
    else:
        st.sidebar.error("❌ Model Evaluation Required")

    # Main content
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        #### 📊 Data Management
        - Upload credit data
        - Clean and preprocess
        - Handle missing values
        - Detect outliers
        """)
    
    with col2:
        st.markdown("""
        #### 🔧 Feature Engineering
        - Create derived features
        - Encode categorical variables
        - Scale numerical features
        - Feature selection
        """)
    
    with col3:
        st.markdown("""
        #### 🤖 Model Development
        - Train multiple algorithms
        - Hyperparameter tuning
        - Model evaluation
        - Bias detection
        """)
    
    # Quick overview if data is available
    if st.session_state.data is not None:
        st.markdown("---")
        st.subheader("📈 Current Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", len(st.session_state.data))
        with col2:
            st.metric("Features", len(st.session_state.data.columns))
        with col3:
            missing_percentage = (st.session_state.data.isnull().sum().sum() / 
                                (st.session_state.data.shape[0] * st.session_state.data.shape[1]) * 100)
            st.metric("Missing Data %", f"{missing_percentage:.1f}%")
        with col4:
            if 'target' in st.session_state.data.columns:
                target_balance = st.session_state.data['target'].value_counts()
                balance_ratio = min(target_balance) / max(target_balance) * 100
                st.metric("Class Balance %", f"{balance_ratio:.1f}%")
    
    # Instructions
    st.markdown("---")
    st.markdown("""
    ### 📋 Getting Started
    
    1. **Data Upload**: Start by uploading your credit dataset (CSV format)
    2. **Data Preprocessing**: Clean and prepare your data for modeling
    3. **Feature Engineering**: Create and select relevant features
    4. **Model Training**: Train and tune multiple classification models
    5. **Model Evaluation**: Assess model performance and select the best one
    6. **Credit Scoring**: Use the trained model for predictions
    
    ### 📝 Data Requirements
    
    Your dataset should include features such as:
    - **Annual Income**: Customer's yearly income
    - **Total Debts**: Total outstanding debt amount
    - **Payment History**: Number of late payments or payment delays
    - **Employment Status**: Current employment situation
    - **Credit Utilization Rate**: Percentage of credit limit used
    - **Demographics**: Age, education level, etc.
    - **Target Variable**: Binary indicator for creditworthiness (0=Bad, 1=Good)
    """)

if __name__ == "__main__":
    main()
