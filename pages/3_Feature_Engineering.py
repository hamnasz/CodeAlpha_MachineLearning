import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from utils.feature_engineer import FeatureEngineer

st.set_page_config(page_title="Feature Engineering", page_icon="⚙️", layout="wide")

def main():
    st.title("⚙️ Feature Engineering")
    
    if st.session_state.processed_data is None:
        st.error("❌ No processed data found. Please complete data preprocessing first.")
        st.info("👈 Go to Data Preprocessing page to clean your data.")
        return
    
    data = st.session_state.processed_data.copy()
    engineer = FeatureEngineer()
    
    # Check for target variable
    if 'target' not in data.columns:
        st.error("❌ Target variable 'target' not found in the dataset.")
        st.info("Please ensure your dataset has a 'target' column representing creditworthiness.")
        return
    
    st.markdown("Create new features and prepare data for machine learning modeling.")
    
    # Feature engineering configuration
    st.subheader("🔧 Feature Engineering Configuration")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Create Features", "Encode Variables", "Scale Features", "Select Features"])
    
    with tab1:
        st.markdown("### 🏗️ Create Derived Features")
        
        # Identify potential features for engineering
        numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        if 'target' in numerical_cols:
            numerical_cols.remove('target')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Financial Ratios:**")
            create_debt_income = st.checkbox("Debt-to-Income Ratio", value=True)
            create_payment_ratio = st.checkbox("Payment-to-Income Ratio", value=True)
            create_utilization_squared = st.checkbox("Credit Utilization Squared", value=True)
            
            st.markdown("**Age-based Features:**")
            create_age_bins = st.checkbox("Age Groups", value=True)
            create_age_income = st.checkbox("Age-Income Interaction", value=True)
        
        with col2:
            st.markdown("**Payment History Features:**")
            create_payment_severity = st.checkbox("Payment Severity Score", value=True)
            create_payment_trend = st.checkbox("Payment Trend", value=False)
            
            st.markdown("**Risk Indicators:**")
            create_risk_score = st.checkbox("Composite Risk Score", value=True)
            create_income_bins = st.checkbox("Income Brackets", value=True)
        
        # Custom feature creation
        st.markdown("**Custom Mathematical Features:**")
        custom_features = st.text_area(
            "Custom features (one per line, e.g., 'log_income = log(annual_income + 1)'):",
            value="log_income = log(annual_income + 1)\nsqrt_age = sqrt(age)",
            help="Create custom features using mathematical expressions"
        )
    
    with tab2:
        st.markdown("### 🔤 Categorical Variable Encoding")
        
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if categorical_cols:
            encoding_method = st.selectbox(
                "Encoding method",
                options=['onehot', 'label', 'target', 'frequency'],
                index=0,
                help="Method for encoding categorical variables"
            )
            
            if encoding_method == 'onehot':
                max_categories = st.slider("Max categories for one-hot encoding", 2, 20, 10)
                drop_first = st.checkbox("Drop first category (avoid multicollinearity)", value=True)
            
            categorical_to_encode = st.multiselect(
                "Select categorical columns to encode",
                options=categorical_cols,
                default=categorical_cols,
                help="Choose which categorical columns to encode"
            )
        else:
            st.info("No categorical columns found in the dataset.")
    
    with tab3:
        st.markdown("### 📏 Feature Scaling")
        
        scaling_method = st.selectbox(
            "Scaling method",
            options=['standard', 'minmax', 'robust', 'none'],
            index=0,
            help="Method for scaling numerical features"
        )
        
        scale_target = st.checkbox("Scale target variable", value=False)
        
        if numerical_cols:
            features_to_scale = st.multiselect(
                "Select features to scale",
                options=numerical_cols,
                default=numerical_cols,
                help="Choose which numerical features to scale"
            )
    
    with tab4:
        st.markdown("### 🎯 Feature Selection")
        
        perform_selection = st.checkbox("Perform feature selection", value=True)
        
        if perform_selection:
            selection_method = st.selectbox(
                "Selection method",
                options=['univariate', 'correlation', 'recursive', 'lasso'],
                index=0,
                help="Method for selecting important features"
            )
            
            if selection_method == 'univariate':
                score_func = st.selectbox(
                    "Scoring function",
                    options=['f_classif', 'mutual_info_classif'],
                    index=0
                )
                k_features = st.slider("Number of features to select", 5, min(20, len(numerical_cols)), 10)
            
            elif selection_method == 'correlation':
                correlation_threshold = st.slider("Correlation threshold", 0.5, 0.95, 0.8, 0.05)
    
    # Engineering execution
    if st.button("🚀 Engineer Features", type="primary"):
        with st.spinner("Engineering features..."):
            try:
                # Configure feature engineering
                config = {
                    'create_debt_income': create_debt_income if 'create_debt_income' in locals() else False,
                    'create_payment_ratio': create_payment_ratio if 'create_payment_ratio' in locals() else False,
                    'create_utilization_squared': create_utilization_squared if 'create_utilization_squared' in locals() else False,
                    'create_age_bins': create_age_bins if 'create_age_bins' in locals() else False,
                    'create_age_income': create_age_income if 'create_age_income' in locals() else False,
                    'create_payment_severity': create_payment_severity if 'create_payment_severity' in locals() else False,
                    'create_risk_score': create_risk_score if 'create_risk_score' in locals() else False,
                    'create_income_bins': create_income_bins if 'create_income_bins' in locals() else False,
                    'custom_features': custom_features if 'custom_features' in locals() else "",
                    'encoding_method': encoding_method if categorical_cols else None,
                    'categorical_to_encode': categorical_to_encode if categorical_cols else [],
                    'scaling_method': scaling_method,
                    'features_to_scale': features_to_scale if numerical_cols else [],
                    'perform_selection': perform_selection if 'perform_selection' in locals() else False,
                    'selection_method': selection_method if perform_selection else None
                }
                
                if 'max_categories' in locals():
                    config['max_categories'] = max_categories
                if 'drop_first' in locals():
                    config['drop_first'] = drop_first
                if 'k_features' in locals():
                    config['k_features'] = k_features
                if 'correlation_threshold' in locals():
                    config['correlation_threshold'] = correlation_threshold
                if 'score_func' in locals():
                    config['score_func'] = score_func
                
                # Engineer features
                engineered_data, engineering_report = engineer.engineer_features(data, config)
                
                # Store engineered data
                st.session_state.engineered_data = engineered_data
                st.session_state.engineering_report = engineering_report
                
                st.success("✅ Feature engineering completed!")
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error during feature engineering: {str(e)}")
    
    # Show results if features have been engineered
    if st.session_state.engineered_data is not None:
        engineered_data = st.session_state.engineered_data
        report = st.session_state.engineering_report
        
        st.markdown("---")
        st.subheader("📊 Feature Engineering Results")
        
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Features",
                len(engineered_data.columns) - 1,  # Exclude target
                delta=len(engineered_data.columns) - len(data.columns)
            )
        
        with col2:
            numerical_features = len(engineered_data.select_dtypes(include=[np.number]).columns) - 1
            st.metric("Numerical Features", numerical_features)
        
        with col3:
            categorical_features = len(engineered_data.select_dtypes(include=['object', 'category']).columns)
            st.metric("Categorical Features", categorical_features)
        
        with col4:
            memory_usage = engineered_data.memory_usage(deep=True).sum() / 1024
            st.metric("Memory Usage (KB)", f"{memory_usage:.1f}")
        
        # Engineering report
        if report:
            st.subheader("📋 Engineering Report")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if 'new_features' in report:
                    st.markdown("**New Features Created:**")
                    for feature in report['new_features']:
                        st.write(f"- {feature}")
                
                if 'encoded_features' in report:
                    st.markdown("**Encoded Features:**")
                    for original, encoded in report['encoded_features'].items():
                        if isinstance(encoded, list):
                            st.write(f"- {original} → {len(encoded)} columns")
                        else:
                            st.write(f"- {original} → {encoded}")
            
            with col2:
                if 'scaled_features' in report:
                    st.markdown("**Scaled Features:**")
                    for feature in report['scaled_features']:
                        st.write(f"- {feature}")
                
                if 'selected_features' in report:
                    st.markdown("**Selected Features:**")
                    st.write(f"Total: {len(report['selected_features'])} features")
                    if 'feature_scores' in report:
                        st.write("Top 5 by importance:")
                        top_features = sorted(report['feature_scores'].items(), 
                                            key=lambda x: x[1], reverse=True)[:5]
                        for feature, score in top_features:
                            st.write(f"- {feature}: {score:.3f}")
        
        # Feature importance analysis
        if 'target' in engineered_data.columns:
            st.subheader("🎯 Feature Importance Analysis")
            
            # Separate features and target
            X = engineered_data.drop('target', axis=1)
            y = engineered_data['target']
            
            # Calculate correlation with target for numerical features
            numerical_features = X.select_dtypes(include=[np.number]).columns
            if len(numerical_features) > 0:
                correlations = X[numerical_features].corrwith(y).abs().sort_values(ascending=False)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Top Correlated Features:**")
                    top_corr = correlations.head(10)
                    for feature, corr in top_corr.items():
                        st.write(f"- {feature}: {corr:.3f}")
                
                with col2:
                    fig = px.bar(
                        x=correlations.head(10).values,
                        y=correlations.head(10).index,
                        orientation='h',
                        title="Top 10 Feature Correlations with Target"
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
        
        # Feature distribution analysis
        st.subheader("📈 Feature Distribution Analysis")
        
        feature_to_analyze = st.selectbox(
            "Select feature to analyze",
            options=[col for col in engineered_data.columns if col != 'target']
        )
        
        if feature_to_analyze:
            col1, col2 = st.columns(2)
            
            with col1:
                if engineered_data[feature_to_analyze].dtype in ['object', 'category']:
                    value_counts = engineered_data[feature_to_analyze].value_counts()
                    fig = px.bar(
                        x=value_counts.index,
                        y=value_counts.values,
                        title=f"Distribution of {feature_to_analyze}"
                    )
                else:
                    fig = px.histogram(
                        engineered_data,
                        x=feature_to_analyze,
                        title=f"Distribution of {feature_to_analyze}",
                        nbins=30
                    )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if 'target' in engineered_data.columns:
                    if engineered_data[feature_to_analyze].dtype in ['object', 'category']:
                        # Grouped bar chart for categorical
                        crosstab = pd.crosstab(engineered_data[feature_to_analyze], 
                                             engineered_data['target'], normalize='index')
                        fig = px.bar(
                            crosstab,
                            title=f"{feature_to_analyze} by Target",
                            barmode='group'
                        )
                    else:
                        # Box plot for numerical
                        fig = px.box(
                            engineered_data,
                            x='target',
                            y=feature_to_analyze,
                            title=f"{feature_to_analyze} by Target"
                        )
                    st.plotly_chart(fig, use_container_width=True)
        
        # Data preview
        st.subheader("👀 Engineered Data Preview")
        st.dataframe(engineered_data.head(10), use_container_width=True)
        
        # Download engineered data
        csv = engineered_data.to_csv(index=False)
        st.download_button(
            label="📥 Download Engineered Data",
            data=csv,
            file_name="engineered_credit_data.csv",
            mime="text/csv"
        )
        
        # Next steps
        st.subheader("➡️ Next Steps")
        st.info("""
        Feature engineering completed! You can now proceed to:
        1. **Model Training** - Train classification models on the engineered features
        2. **Model Evaluation** - Assess model performance and interpretability
        """)
    
    else:
        # Show current data status for feature engineering
        st.subheader("📊 Current Data Analysis")
        
        # Feature distribution overview
        numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Numerical Features", len(numerical_cols))
        with col2:
            st.metric("Categorical Features", len(categorical_cols))
        with col3:
            st.metric("Total Features", len(data.columns) - 1)  # Exclude target
        
        # Potential feature engineering opportunities
        st.markdown("### 🔍 Feature Engineering Opportunities")
        
        opportunities = []
        
        # Check for ratio features
        if any('income' in col.lower() for col in numerical_cols) and any('debt' in col.lower() for col in numerical_cols):
            opportunities.append("✅ Debt-to-Income ratio can be created")
        
        if any('age' in col.lower() for col in numerical_cols):
            opportunities.append("✅ Age-based groupings can be created")
        
        if any('utilization' in col.lower() for col in numerical_cols):
            opportunities.append("✅ Credit utilization transformations possible")
        
        if len(categorical_cols) > 0:
            opportunities.append("✅ Categorical encoding needed")
        
        if len(numerical_cols) > 2:
            opportunities.append("✅ Feature scaling recommended")
        
        for opportunity in opportunities:
            st.write(opportunity)
        
        if not opportunities:
            st.info("Configure feature engineering options above and click 'Engineer Features' to proceed.")

if __name__ == "__main__":
    main()
