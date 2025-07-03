import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Data Upload", page_icon="📊", layout="wide")

def create_sample_dataset():
    """Create a sample credit dataset for testing purposes"""
    np.random.seed(42)
    n_samples = 1000
    
    # Create realistic credit data
    data = {
        'age': np.random.randint(18, 80, n_samples),
        'income': np.random.normal(50000, 20000, n_samples).clip(min=15000),
        'employment_length': np.random.randint(0, 40, n_samples),
        'loan_amount': np.random.normal(25000, 15000, n_samples).clip(min=1000),
        'credit_score': np.random.randint(300, 850, n_samples),
        'debt_to_income': np.random.uniform(0.1, 0.8, n_samples),
        'payment_history': np.random.choice(['excellent', 'good', 'fair', 'poor'], n_samples, p=[0.3, 0.4, 0.2, 0.1]),
        'num_credit_accounts': np.random.randint(1, 15, n_samples),
        'home_ownership': np.random.choice(['own', 'rent', 'mortgage'], n_samples, p=[0.3, 0.4, 0.3]),
        'education': np.random.choice(['high_school', 'bachelor', 'master', 'phd'], n_samples, p=[0.4, 0.35, 0.2, 0.05]),
        'marital_status': np.random.choice(['single', 'married', 'divorced'], n_samples, p=[0.4, 0.5, 0.1]),
    }
    
    # Create target variable based on realistic criteria
    risk_score = (
        (data['credit_score'] / 850) * 0.4 +
        (1 - data['debt_to_income']) * 0.3 +
        (data['income'] / 100000) * 0.2 +
        (data['employment_length'] / 40) * 0.1
    )
    
    # Add some noise and convert to binary
    risk_score += np.random.normal(0, 0.1, n_samples)
    data['default_risk'] = (risk_score < 0.5).astype(int)
    
    return pd.DataFrame(data)

def main():
    st.title("📊 Data Upload & Initial Analysis")
    
    st.markdown("""
    Upload your credit dataset to begin the modeling process. The system accepts CSV files with credit-related features.
    """)
    
    # Add sample data option
    st.info("💡 **Tip**: If you're having upload issues, you can create a sample dataset to test the platform.")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        if st.button("🎯 Create Sample Dataset"):
            sample_data = create_sample_dataset()
            st.session_state.data = sample_data
            st.success("✅ Sample dataset created successfully!")
            st.rerun()
    
    with col1:
        st.markdown("**Upload your own data:**")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type="csv",
        help="Upload a CSV file containing credit data with features like income, debts, payment history, etc.",
        accept_multiple_files=False
    )
    
    if uploaded_file is not None:
        try:
            # Check file size
            file_size = uploaded_file.size
            if file_size > 200 * 1024 * 1024:  # 200MB limit
                st.error("File size exceeds 200MB limit. Please upload a smaller file.")
                return
            
            # Load data with error handling
            try:
                data = pd.read_csv(uploaded_file)
            except pd.errors.EmptyDataError:
                st.error("The uploaded file is empty. Please upload a valid CSV file.")
                return
            except pd.errors.ParserError as e:
                st.error(f"Error parsing CSV file: {str(e)}")
                return
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
                return
            
            st.session_state.data = data
            st.success("✅ Data uploaded successfully!")
            
            # Basic information
            st.subheader("📋 Dataset Overview")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Records", len(data))
            with col2:
                st.metric("Total Features", len(data.columns))
            with col3:
                st.metric("Memory Usage", f"{data.memory_usage(deep=True).sum() / 1024:.1f} KB")
            with col4:
                duplicates = data.duplicated().sum()
                st.metric("Duplicate Rows", duplicates)
            
            # Display first few rows
            st.subheader("🔍 Data Preview")
            st.dataframe(data.head(10), use_container_width=True)
            
            # Data types and missing values analysis
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Data Types")
                dtype_info = pd.DataFrame({
                    'Column': data.columns,
                    'Data Type': data.dtypes.astype(str),
                    'Non-Null Count': data.count(),
                    'Null Count': data.isnull().sum(),
                    'Null Percentage': (data.isnull().sum() / len(data) * 100).round(2)
                })
                st.dataframe(dtype_info, use_container_width=True)
            
            with col2:
                st.subheader("📈 Missing Data Visualization")
                missing_data = data.isnull().sum()
                missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
                
                if len(missing_data) > 0:
                    fig = px.bar(
                        x=missing_data.values,
                        y=missing_data.index,
                        orientation='h',
                        title="Missing Values by Column",
                        labels={'x': 'Number of Missing Values', 'y': 'Columns'}
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("🎉 No missing values detected!")
            
            # Statistical summary
            st.subheader("📊 Statistical Summary")
            
            # Separate numerical and categorical columns
            numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if numerical_cols:
                st.markdown("**Numerical Features:**")
                st.dataframe(data[numerical_cols].describe(), use_container_width=True)
            
            if categorical_cols:
                st.markdown("**Categorical Features:**")
                for col in categorical_cols:
                    value_counts = data[col].value_counts()
                    st.write(f"**{col}**: {len(value_counts)} unique values")
                    if len(value_counts) <= 10:
                        st.write(value_counts.to_dict())
                    else:
                        st.write(f"Top 5 values: {value_counts.head().to_dict()}")
            
            # Target variable analysis (if exists)
            st.subheader("🎯 Target Variable Analysis")
            
            # Try to identify target column
            potential_targets = ['target', 'label', 'class', 'default', 'creditworthy', 'good_bad']
            target_col = None
            
            for col in potential_targets:
                if col in data.columns:
                    target_col = col
                    break
            
            if target_col is None:
                # Let user select target column
                target_col = st.selectbox(
                    "Select the target variable (creditworthiness indicator):",
                    options=['None'] + list(data.columns),
                    help="Choose the column that represents creditworthiness (0=Bad, 1=Good)"
                )
                
                if target_col != 'None':
                    # Update the column name to 'target' for consistency
                    if target_col != 'target':
                        data = data.rename(columns={target_col: 'target'})
                        st.session_state.data = data
                        st.info(f"Renamed '{target_col}' to 'target' for consistency")
                        target_col = 'target'
            
            if target_col and target_col in data.columns:
                target_dist = data[target_col].value_counts().sort_index()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Target Distribution:**")
                    for value, count in target_dist.items():
                        percentage = count / len(data) * 100
                        st.write(f"Class {value}: {count} ({percentage:.1f}%)")
                    
                    # Class balance warning
                    balance_ratio = min(target_dist) / max(target_dist)
                    if balance_ratio < 0.3:
                        st.warning(f"⚠️ Class imbalance detected! Ratio: {balance_ratio:.2f}")
                    else:
                        st.success(f"✅ Reasonable class balance. Ratio: {balance_ratio:.2f}")
                
                with col2:
                    fig = px.pie(
                        values=target_dist.values,
                        names=target_dist.index,
                        title="Target Variable Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Correlation heatmap for numerical features
            if len(numerical_cols) > 1:
                st.subheader("🔗 Feature Correlation Matrix")
                
                correlation_matrix = data[numerical_cols].corr()
                
                fig = px.imshow(
                    correlation_matrix,
                    text_auto=True,
                    aspect="auto",
                    title="Correlation Matrix of Numerical Features",
                    color_continuous_scale="RdBu_r"
                )
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
                
                # High correlation warning
                high_corr_pairs = []
                for i in range(len(correlation_matrix.columns)):
                    for j in range(i+1, len(correlation_matrix.columns)):
                        corr_value = abs(correlation_matrix.iloc[i, j])
                        if corr_value > 0.8:
                            high_corr_pairs.append((
                                correlation_matrix.columns[i],
                                correlation_matrix.columns[j],
                                corr_value
                            ))
                
                if high_corr_pairs:
                    st.warning("⚠️ High correlation detected between features:")
                    for col1, col2, corr in high_corr_pairs:
                        st.write(f"- {col1} ↔ {col2}: {corr:.3f}")
            
            # Data quality assessment
            st.subheader("🔍 Data Quality Assessment")
            
            quality_issues = []
            
            # Check for missing values
            missing_percentage = (data.isnull().sum().sum() / (data.shape[0] * data.shape[1]) * 100)
            if missing_percentage > 5:
                quality_issues.append(f"High missing data: {missing_percentage:.1f}%")
            
            # Check for duplicates
            if duplicates > 0:
                quality_issues.append(f"Duplicate rows: {duplicates}")
            
            # Check for constant columns
            constant_cols = [col for col in data.columns if data[col].nunique() <= 1]
            if constant_cols:
                quality_issues.append(f"Constant columns: {constant_cols}")
            
            # Check for high cardinality categorical columns
            high_cardinality_cols = [col for col in categorical_cols if data[col].nunique() > 50]
            if high_cardinality_cols:
                quality_issues.append(f"High cardinality categorical columns: {high_cardinality_cols}")
            
            if quality_issues:
                st.warning("⚠️ Data Quality Issues Detected:")
                for issue in quality_issues:
                    st.write(f"- {issue}")
            else:
                st.success("✅ No major data quality issues detected!")
            
            # Next steps
            st.subheader("➡️ Next Steps")
            st.info("""
            Data successfully loaded! You can now proceed to:
            1. **Data Preprocessing** - Clean and prepare the data
            2. **Feature Engineering** - Create and select relevant features
            3. **Model Training** - Train classification models
            """)
            
        except Exception as e:
            st.error(f"❌ Error loading data: {str(e)}")
            st.info("Please ensure your CSV file is properly formatted and accessible.")
    
    else:
        st.info("👆 Please upload a CSV file to begin the analysis.")
        
        # Example data format
        st.subheader("📋 Expected Data Format")
        st.markdown("""
        Your CSV file should contain columns similar to:
        
        | annual_income | total_debts | payment_history | employment_status | credit_utilization | age | education | target |
        |---------------|-------------|-----------------|-------------------|-------------------|-----|-----------|--------|
        | 50000 | 15000 | 2 | Employed | 0.3 | 35 | Bachelor | 1 |
        | 30000 | 25000 | 5 | Unemployed | 0.8 | 28 | High School | 0 |
        
        Where:
        - **target**: 1 = Good credit, 0 = Bad credit
        - **payment_history**: Number of late payments
        - **credit_utilization**: Ratio between 0 and 1
        """)

if __name__ == "__main__":
    main()
