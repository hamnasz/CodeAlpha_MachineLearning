import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.data_processor import DataProcessor

st.set_page_config(page_title="Data Preprocessing", page_icon="🧹", layout="wide")

def main():
    st.title("🧹 Data Preprocessing")
    
    if st.session_state.data is None:
        st.error("❌ No data found. Please upload data first.")
        st.info("👈 Go to Data Upload page to load your dataset.")
        return
    
    data = st.session_state.data.copy()
    processor = DataProcessor()
    
    st.markdown("Clean and prepare your data for modeling by handling missing values, outliers, and data types.")
    
    # Preprocessing options
    st.subheader("🔧 Preprocessing Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Missing Value Handling:**")
        missing_strategy_num = st.selectbox(
            "Numerical features",
            options=['mean', 'median', 'mode', 'drop', 'forward_fill'],
            index=1,
            help="Strategy for handling missing numerical values"
        )
        
        missing_strategy_cat = st.selectbox(
            "Categorical features",
            options=['mode', 'constant', 'drop'],
            index=0,
            help="Strategy for handling missing categorical values"
        )
        
        if missing_strategy_cat == 'constant':
            missing_constant = st.text_input("Fill value for categorical", value="Unknown")
        else:
            missing_constant = None
    
    with col2:
        st.markdown("**Outlier Detection:**")
        outlier_method = st.selectbox(
            "Detection method",
            options=['iqr', 'zscore', 'isolation_forest', 'none'],
            index=0,
            help="Method for detecting outliers"
        )
        
        if outlier_method == 'iqr':
            iqr_threshold = st.slider("IQR multiplier", 1.0, 3.0, 1.5, 0.1)
        elif outlier_method == 'zscore':
            zscore_threshold = st.slider("Z-score threshold", 2.0, 4.0, 3.0, 0.1)
        else:
            iqr_threshold = zscore_threshold = None
        
        outlier_action = st.selectbox(
            "Outlier handling",
            options=['remove', 'cap', 'keep'],
            index=1,
            help="Action to take for detected outliers"
        )
    
    # Additional preprocessing options
    st.markdown("**Additional Options:**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        remove_duplicates = st.checkbox("Remove duplicate rows", value=True)
    with col2:
        remove_constant = st.checkbox("Remove constant columns", value=True)
    with col3:
        standardize_names = st.checkbox("Standardize column names", value=True)
    
    # Process data button
    if st.button("🚀 Process Data", type="primary"):
        with st.spinner("Processing data..."):
            try:
                # Configure processor
                config = {
                    'missing_strategy_num': missing_strategy_num,
                    'missing_strategy_cat': missing_strategy_cat,
                    'missing_constant': missing_constant,
                    'outlier_method': outlier_method,
                    'outlier_action': outlier_action,
                    'remove_duplicates': remove_duplicates,
                    'remove_constant': remove_constant,
                    'standardize_names': standardize_names
                }
                
                if outlier_method == 'iqr':
                    config['iqr_threshold'] = iqr_threshold
                elif outlier_method == 'zscore':
                    config['zscore_threshold'] = zscore_threshold
                
                # Process data
                processed_data, processing_report = processor.preprocess_data(data, config)
                
                # Store processed data
                st.session_state.processed_data = processed_data
                st.session_state.processing_report = processing_report
                
                st.success("✅ Data preprocessing completed!")
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error during preprocessing: {str(e)}")
    
    # Show results if data has been processed
    if st.session_state.processed_data is not None:
        processed_data = st.session_state.processed_data
        report = st.session_state.processing_report
        
        st.markdown("---")
        st.subheader("📊 Preprocessing Results")
        
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Records",
                len(processed_data),
                delta=len(processed_data) - len(data)
            )
        
        with col2:
            st.metric(
                "Features",
                len(processed_data.columns),
                delta=len(processed_data.columns) - len(data.columns)
            )
        
        with col3:
            missing_before = data.isnull().sum().sum()
            missing_after = processed_data.isnull().sum().sum()
            st.metric(
                "Missing Values",
                missing_after,
                delta=missing_after - missing_before
            )
        
        with col4:
            memory_before = data.memory_usage(deep=True).sum() / 1024
            memory_after = processed_data.memory_usage(deep=True).sum() / 1024
            st.metric(
                "Memory (KB)",
                f"{memory_after:.1f}",
                delta=f"{memory_after - memory_before:.1f}"
            )
        
        # Processing report
        if report:
            st.subheader("📋 Processing Report")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if 'missing_handled' in report:
                    st.markdown("**Missing Values Handled:**")
                    for col, info in report['missing_handled'].items():
                        st.write(f"- {col}: {info['count']} values filled with {info['strategy']}")
                
                if 'outliers_detected' in report:
                    st.markdown("**Outliers Detected:**")
                    for col, count in report['outliers_detected'].items():
                        st.write(f"- {col}: {count} outliers {report.get('outlier_action', 'detected')}")
            
            with col2:
                if 'duplicates_removed' in report:
                    st.markdown(f"**Duplicates Removed:** {report['duplicates_removed']}")
                
                if 'constant_columns_removed' in report:
                    st.markdown("**Constant Columns Removed:**")
                    for col in report['constant_columns_removed']:
                        st.write(f"- {col}")
                
                if 'columns_renamed' in report:
                    st.markdown("**Columns Renamed:**")
                    for old, new in report['columns_renamed'].items():
                        st.write(f"- {old} → {new}")
        
        # Before/After comparison
        st.subheader("🔄 Before vs After Comparison")
        
        tab1, tab2, tab3 = st.tabs(["Data Types", "Missing Values", "Statistics"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Before Processing:**")
                before_types = pd.DataFrame({
                    'Column': data.columns,
                    'Type': data.dtypes.astype(str),
                    'Non-Null': data.count()
                })
                st.dataframe(before_types, use_container_width=True)
            
            with col2:
                st.markdown("**After Processing:**")
                after_types = pd.DataFrame({
                    'Column': processed_data.columns,
                    'Type': processed_data.dtypes.astype(str),
                    'Non-Null': processed_data.count()
                })
                st.dataframe(after_types, use_container_width=True)
        
        with tab2:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Before Processing:**")
                missing_before = data.isnull().sum()
                missing_before = missing_before[missing_before > 0]
                if len(missing_before) > 0:
                    fig1 = px.bar(
                        x=missing_before.values,
                        y=missing_before.index,
                        orientation='h',
                        title="Missing Values Before"
                    )
                    st.plotly_chart(fig1, use_container_width=True)
                else:
                    st.info("No missing values")
            
            with col2:
                st.markdown("**After Processing:**")
                missing_after = processed_data.isnull().sum()
                missing_after = missing_after[missing_after > 0]
                if len(missing_after) > 0:
                    fig2 = px.bar(
                        x=missing_after.values,
                        y=missing_after.index,
                        orientation='h',
                        title="Missing Values After"
                    )
                    st.plotly_chart(fig2, use_container_width=True)
                else:
                    st.success("✅ No missing values!")
        
        with tab3:
            numerical_cols = processed_data.select_dtypes(include=[np.number]).columns
            if len(numerical_cols) > 0:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Before Processing:**")
                    st.dataframe(data[numerical_cols].describe(), use_container_width=True)
                
                with col2:
                    st.markdown("**After Processing:**")
                    st.dataframe(processed_data[numerical_cols].describe(), use_container_width=True)
        
        # Data preview
        st.subheader("👀 Processed Data Preview")
        st.dataframe(processed_data.head(10), use_container_width=True)
        
        # Download processed data
        csv = processed_data.to_csv(index=False)
        st.download_button(
            label="📥 Download Processed Data",
            data=csv,
            file_name="processed_credit_data.csv",
            mime="text/csv"
        )
        
        # Next steps
        st.subheader("➡️ Next Steps")
        st.info("""
        Data preprocessing completed! You can now proceed to:
        1. **Feature Engineering** - Create derived features and prepare for modeling
        2. **Model Training** - Train classification models on the clean data
        """)
    
    else:
        # Show current data status
        st.subheader("📊 Current Data Status")
        
        # Missing values analysis
        missing_data = data.isnull().sum()
        missing_data = missing_data[missing_data > 0]
        
        if len(missing_data) > 0:
            st.markdown("**Missing Values Found:**")
            col1, col2 = st.columns(2)
            
            with col1:
                missing_df = pd.DataFrame({
                    'Column': missing_data.index,
                    'Missing Count': missing_data.values,
                    'Missing %': (missing_data.values / len(data) * 100).round(2)
                })
                st.dataframe(missing_df, use_container_width=True)
            
            with col2:
                fig = px.bar(
                    x=missing_data.values,
                    y=missing_data.index,
                    orientation='h',
                    title="Missing Values by Column"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Outlier preview (for numerical columns)
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            st.markdown("**Potential Outliers Preview:**")
            
            selected_col = st.selectbox("Select column for outlier analysis", numerical_cols)
            
            if selected_col:
                col_data = data[selected_col].dropna()
                
                # Calculate outlier bounds
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Potential Outliers", len(outliers))
                    st.metric("Outlier %", f"{len(outliers)/len(col_data)*100:.1f}%")
                
                with col2:
                    fig = px.box(y=col_data, title=f"Box Plot - {selected_col}")
                    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
