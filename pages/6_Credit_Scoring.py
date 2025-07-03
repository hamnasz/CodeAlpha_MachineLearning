import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
import json
from datetime import datetime

st.set_page_config(page_title="Credit Scoring", page_icon="🎯", layout="wide")

def main():
    st.title("🎯 Credit Scoring & Predictions")
    
    if not st.session_state.trained_models:
        st.error("❌ No trained models found. Please complete model training first.")
        st.info("👈 Go to Model Training page to train your models.")
        return
    
    st.markdown("Use your trained models to assess creditworthiness and make lending decisions.")
    
    # Model selection
    st.subheader("🤖 Model Selection")
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_model_name = st.selectbox(
            "Choose model for scoring",
            options=list(st.session_state.trained_models.keys()),
            index=list(st.session_state.trained_models.keys()).index(st.session_state.best_model) 
            if st.session_state.best_model in st.session_state.trained_models else 0
        )
    
    with col2:
        scoring_threshold = st.slider(
            "Approval threshold",
            min_value=0.1,
            max_value=0.9,
            value=0.5,
            step=0.05,
            help="Probability threshold for loan approval"
        )
    
    selected_model = st.session_state.trained_models[selected_model_name]
    model_results = st.session_state.model_results[selected_model_name]
    
    # Model performance summary
    with st.expander("📊 Selected Model Performance"):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", f"{model_results['test_accuracy']:.3f}")
        with col2:
            st.metric("Precision", f"{model_results['test_precision']:.3f}")
        with col3:
            st.metric("Recall", f"{model_results['test_recall']:.3f}")
        with col4:
            st.metric("ROC-AUC", f"{model_results['test_roc_auc']:.3f}")
    
    # Scoring options
    st.subheader("🔢 Scoring Options")
    
    scoring_mode = st.radio(
        "Choose scoring mode",
        options=["Single Applicant", "Batch Scoring", "Model Comparison"],
        horizontal=True
    )
    
    if scoring_mode == "Single Applicant":
        single_applicant_scoring(selected_model, selected_model_name, scoring_threshold)
    
    elif scoring_mode == "Batch Scoring":
        batch_scoring(selected_model, selected_model_name, scoring_threshold)
    
    elif scoring_mode == "Model Comparison":
        model_comparison_scoring(scoring_threshold)

def single_applicant_scoring(model, model_name, threshold):
    """Handle single applicant credit scoring"""
    
    st.markdown("### 👤 Single Applicant Assessment")
    
    # Get feature names from the engineered data
    if st.session_state.engineered_data is not None:
        feature_names = [col for col in st.session_state.engineered_data.columns if col != 'target']
    else:
        st.error("No engineered data available for feature reference.")
        return
    
    # Create input form
    st.markdown("**Enter applicant information:**")
    
    # Organize features into categories for better UX
    numerical_features = []
    categorical_features = []
    
    if st.session_state.engineered_data is not None:
        sample_data = st.session_state.engineered_data.drop('target', axis=1)
        numerical_features = sample_data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = sample_data.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Input form in columns
    col1, col2 = st.columns(2)
    
    applicant_data = {}
    
    with col1:
        st.markdown("**📊 Financial Information:**")
        
        # Common financial features
        financial_features = [f for f in numerical_features if any(keyword in f.lower() 
                            for keyword in ['income', 'debt', 'salary', 'amount', 'balance', 'payment'])]
        
        for feature in financial_features[:6]:  # Limit to first 6 for better UX
            if 'ratio' in feature.lower() or 'rate' in feature.lower():
                default_val = 0.5
                min_val = 0.0
                max_val = 1.0
                step = 0.01
            elif 'income' in feature.lower():
                default_val = 50000.0
                min_val = 0.0
                max_val = 500000.0
                step = 1000.0
            elif 'debt' in feature.lower():
                default_val = 15000.0
                min_val = 0.0
                max_val = 200000.0
                step = 1000.0
            elif 'payment' in feature.lower():
                default_val = 0.0
                min_val = 0.0
                max_val = 50.0
                step = 1.0
            else:
                default_val = 0.0
                min_val = 0.0
                max_val = 100000.0
                step = 100.0
            
            applicant_data[feature] = st.number_input(
                feature.replace('_', ' ').title(),
                value=default_val,
                min_value=min_val,
                max_value=max_val,
                step=step
            )
    
    with col2:
        st.markdown("**👤 Personal Information:**")
        
        # Personal and demographic features
        personal_features = [f for f in numerical_features if any(keyword in f.lower() 
                           for keyword in ['age', 'year', 'time', 'duration', 'length'])]
        
        for feature in personal_features[:6]:  # Limit to first 6
            if 'age' in feature.lower():
                default_val = 35
                min_val = 18
                max_val = 80
                step = 1
            elif 'year' in feature.lower():
                default_val = 5
                min_val = 0
                max_val = 40
                step = 1
            else:
                default_val = 0
                min_val = 0
                max_val = 100
                step = 1
            
            applicant_data[feature] = st.number_input(
                feature.replace('_', ' ').title(),
                value=default_val,
                min_value=min_val,
                max_value=max_val,
                step=step
            )
        
        # Categorical features
        if categorical_features:
            st.markdown("**📝 Additional Information:**")
            for feature in categorical_features[:3]:  # Limit to first 3
                unique_values = st.session_state.engineered_data[feature].unique()
                applicant_data[feature] = st.selectbox(
                    feature.replace('_', ' ').title(),
                    options=unique_values
                )
    
    # Handle remaining features
    remaining_features = [f for f in feature_names if f not in applicant_data]
    
    if remaining_features:
        with st.expander(f"🔧 Additional Features ({len(remaining_features)} features)"):
            cols = st.columns(3)
            for i, feature in enumerate(remaining_features):
                with cols[i % 3]:
                    if feature in numerical_features:
                        # Get some stats from the data for better defaults
                        feature_data = st.session_state.engineered_data[feature]
                        default_val = float(feature_data.median())
                        min_val = float(feature_data.min())
                        max_val = float(feature_data.max())
                        
                        applicant_data[feature] = st.number_input(
                            feature.replace('_', ' ').title(),
                            value=default_val,
                            min_value=min_val,
                            max_value=max_val,
                            key=f"remaining_{feature}"
                        )
                    else:
                        unique_values = st.session_state.engineered_data[feature].unique()
                        applicant_data[feature] = st.selectbox(
                            feature.replace('_', ' ').title(),
                            options=unique_values,
                            key=f"remaining_{feature}"
                        )
    
    # Make prediction
    if st.button("🎯 Assess Credit Risk", type="primary"):
        try:
            # Prepare input data
            input_df = pd.DataFrame([applicant_data])
            
            # Ensure all features are present and in correct order
            for feature in feature_names:
                if feature not in input_df.columns:
                    # Use median for missing numerical features, mode for categorical
                    if feature in numerical_features:
                        input_df[feature] = st.session_state.engineered_data[feature].median()
                    else:
                        input_df[feature] = st.session_state.engineered_data[feature].mode()[0]
            
            # Reorder columns to match training data
            input_df = input_df[feature_names]
            
            # Make prediction
            prediction = model.predict(input_df)[0]
            
            if hasattr(model, 'predict_proba'):
                probability = model.predict_proba(input_df)[0, 1]
            else:
                probability = prediction
            
            # Display results
            st.markdown("---")
            st.subheader("📋 Credit Assessment Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Credit score (scale probability to 300-850 range)
                credit_score = int(300 + (probability * 550))
                st.metric("Credit Score", credit_score)
            
            with col2:
                st.metric("Default Probability", f"{probability:.1%}")
            
            with col3:
                approval_status = "APPROVED" if probability >= threshold else "REJECTED"
                status_color = "green" if approval_status == "APPROVED" else "red"
                st.markdown(f"**Decision:** <span style='color:{status_color}'>{approval_status}</span>", 
                           unsafe_allow_html=True)
            
            # Risk assessment gauge
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = probability * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Default Risk (%)"},
                delta = {'reference': threshold * 100},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 20], 'color': "lightgreen"},
                        {'range': [20, 50], 'color': "yellow"},
                        {'range': [50, 80], 'color': "orange"},
                        {'range': [80, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': threshold * 100
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature contribution analysis
            if 'feature_importance' in model_results:
                st.subheader("🔍 Key Factors in Decision")
                
                importance = model_results['feature_importance']
                
                # Get top contributing features
                applicant_values = input_df.iloc[0]
                
                # Calculate feature contributions (simplified)
                contributions = []
                for feature, imp in importance.items():
                    if feature in applicant_values.index:
                        value = applicant_values[feature]
                        # Normalize value and multiply by importance
                        if feature in numerical_features:
                            feature_data = st.session_state.engineered_data[feature]
                            normalized_value = (value - feature_data.mean()) / feature_data.std()
                        else:
                            normalized_value = 1.0  # For categorical features
                        
                        contribution = imp * normalized_value
                        contributions.append({
                            'Feature': feature,
                            'Value': value,
                            'Importance': imp,
                            'Contribution': contribution
                        })
                
                # Sort by absolute contribution
                contributions.sort(key=lambda x: abs(x['Contribution']), reverse=True)
                
                # Display top factors
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Positive Factors (Reduce Risk):**")
                    positive_factors = [c for c in contributions[:5] if c['Contribution'] < 0]
                    for factor in positive_factors:
                        st.write(f"• {factor['Feature'].replace('_', ' ').title()}: {factor['Value']}")
                
                with col2:
                    st.markdown("**Risk Factors (Increase Risk):**")
                    risk_factors = [c for c in contributions[:5] if c['Contribution'] > 0]
                    for factor in risk_factors:
                        st.write(f"• {factor['Feature'].replace('_', ' ').title()}: {factor['Value']}")
            
            # Recommendations
            st.subheader("💡 Recommendations")
            
            if approval_status == "APPROVED":
                st.success("✅ **Recommendation: APPROVE LOAN**")
                st.write("• Low default risk identified")
                st.write("• Monitor for any changes in financial situation")
                if probability > 0.3:
                    st.write("• Consider higher interest rate due to moderate risk")
            else:
                st.error("❌ **Recommendation: REJECT LOAN**")
                st.write("• High default risk identified")
                st.write("• Consider alternative products or co-signer")
                st.write("• Review application in 6-12 months")
            
            # Save prediction log
            prediction_log = {
                'timestamp': datetime.now().isoformat(),
                'model_used': model_name,
                'threshold': threshold,
                'prediction': int(prediction),
                'probability': float(probability),
                'credit_score': credit_score,
                'decision': approval_status,
                'applicant_data': applicant_data
            }
            
            # Store in session state for batch processing
            if 'prediction_history' not in st.session_state:
                st.session_state.prediction_history = []
            st.session_state.prediction_history.append(prediction_log)
            
        except Exception as e:
            st.error(f"❌ Error making prediction: {str(e)}")
            st.info("Please ensure all required fields are filled correctly.")

def batch_scoring(model, model_name, threshold):
    """Handle batch credit scoring"""
    
    st.markdown("### 📊 Batch Credit Scoring")
    
    # File upload for batch scoring
    uploaded_file = st.file_uploader(
        "Upload CSV file for batch scoring",
        type="csv",
        help="Upload a CSV file with the same features as your training data"
    )
    
    if uploaded_file is not None:
        try:
            # Load batch data
            batch_data = pd.read_csv(uploaded_file)
            
            st.success(f"✅ Loaded {len(batch_data)} records for scoring")
            
            # Show data preview
            with st.expander("👀 Data Preview"):
                st.dataframe(batch_data.head(10), use_container_width=True)
            
            # Check feature compatibility
            required_features = [col for col in st.session_state.engineered_data.columns if col != 'target']
            missing_features = [f for f in required_features if f not in batch_data.columns]
            extra_features = [f for f in batch_data.columns if f not in required_features]
            
            if missing_features:
                st.warning(f"⚠️ Missing features: {missing_features}")
                st.info("Missing features will be filled with median/mode values from training data.")
            
            if extra_features:
                st.info(f"ℹ️ Extra features will be ignored: {extra_features}")
            
            # Score batch button
            if st.button("🚀 Score Batch", type="primary"):
                with st.spinner("Scoring batch data..."):
                    try:
                        # Prepare batch data
                        scoring_data = batch_data.copy()
                        
                        # Handle missing features
                        for feature in required_features:
                            if feature not in scoring_data.columns:
                                if feature in st.session_state.engineered_data.select_dtypes(include=[np.number]).columns:
                                    scoring_data[feature] = st.session_state.engineered_data[feature].median()
                                else:
                                    scoring_data[feature] = st.session_state.engineered_data[feature].mode()[0]
                        
                        # Select and reorder features
                        scoring_data = scoring_data[required_features]
                        
                        # Make predictions
                        predictions = model.predict(scoring_data)
                        
                        if hasattr(model, 'predict_proba'):
                            probabilities = model.predict_proba(scoring_data)[:, 1]
                        else:
                            probabilities = predictions.astype(float)
                        
                        # Create results dataframe
                        results_df = batch_data.copy()
                        results_df['Prediction'] = predictions
                        results_df['Default_Probability'] = probabilities
                        results_df['Credit_Score'] = (300 + (probabilities * 550)).astype(int)
                        results_df['Decision'] = ['APPROVED' if p >= threshold else 'REJECTED' 
                                                for p in probabilities]
                        
                        # Display results summary
                        st.subheader("📊 Batch Scoring Results")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Total Scored", len(results_df))
                        with col2:
                            approved_count = sum(results_df['Decision'] == 'APPROVED')
                            st.metric("Approved", approved_count)
                        with col3:
                            approval_rate = approved_count / len(results_df) * 100
                            st.metric("Approval Rate", f"{approval_rate:.1f}%")
                        with col4:
                            avg_score = results_df['Credit_Score'].mean()
                            st.metric("Avg Credit Score", f"{avg_score:.0f}")
                        
                        # Results visualization
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Decision distribution
                            decision_counts = results_df['Decision'].value_counts()
                            fig = px.pie(values=decision_counts.values, names=decision_counts.index,
                                       title="Approval Decision Distribution")
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            # Credit score distribution
                            fig = px.histogram(results_df, x='Credit_Score', nbins=20,
                                             title="Credit Score Distribution")
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Risk distribution
                        st.subheader("📈 Risk Analysis")
                        
                        # Create risk buckets
                        results_df['Risk_Bucket'] = pd.cut(results_df['Default_Probability'], 
                                                         bins=5, 
                                                         labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
                        
                        risk_summary = results_df.groupby('Risk_Bucket').agg({
                            'Default_Probability': ['count', 'mean'],
                            'Credit_Score': 'mean'
                        }).round(3)
                        
                        risk_summary.columns = ['Count', 'Avg_Default_Prob', 'Avg_Credit_Score']
                        st.dataframe(risk_summary, use_container_width=True)
                        
                        # Results table
                        st.subheader("📋 Detailed Results")
                        
                        # Display options
                        show_all = st.checkbox("Show all columns")
                        if show_all:
                            display_df = results_df
                        else:
                            # Show key columns only
                            key_columns = ['Prediction', 'Default_Probability', 'Credit_Score', 'Decision']
                            available_key_cols = [col for col in key_columns if col in results_df.columns]
                            
                            # Add first few original columns
                            original_cols = batch_data.columns.tolist()[:3]
                            display_columns = original_cols + available_key_cols
                            display_df = results_df[display_columns]
                        
                        st.dataframe(display_df, use_container_width=True)
                        
                        # Download results
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Batch Results",
                            data=csv,
                            file_name=f"batch_scoring_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                        
                    except Exception as e:
                        st.error(f"❌ Error during batch scoring: {str(e)}")
        
        except Exception as e:
            st.error(f"❌ Error loading batch file: {str(e)}")
    
    else:
        st.info("👆 Upload a CSV file to begin batch scoring")
        
        # Show expected format
        if st.session_state.engineered_data is not None:
            st.subheader("📋 Expected File Format")
            
            sample_features = [col for col in st.session_state.engineered_data.columns if col != 'target'][:5]
            sample_data = st.session_state.engineered_data[sample_features].head(3)
            
            st.markdown("**Your CSV should contain columns like:**")
            st.dataframe(sample_data, use_container_width=True)

def model_comparison_scoring(threshold):
    """Compare predictions across all trained models"""
    
    st.markdown("### 🔬 Model Comparison")
    
    if len(st.session_state.trained_models) < 2:
        st.warning("⚠️ Need at least 2 trained models for comparison")
        return
    
    # Single applicant comparison
    st.markdown("**Enter applicant data for model comparison:**")
    
    # Simplified input for comparison
    col1, col2, col3 = st.columns(3)
    
    # Get some key features for simplified input
    if st.session_state.engineered_data is not None:
        numerical_features = st.session_state.engineered_data.select_dtypes(include=[np.number]).columns.tolist()
        if 'target' in numerical_features:
            numerical_features.remove('target')
        
        key_features = numerical_features[:6]  # Take first 6 numerical features
        
        comparison_data = {}
        
        for i, feature in enumerate(key_features):
            with [col1, col2, col3][i % 3]:
                feature_data = st.session_state.engineered_data[feature]
                comparison_data[feature] = st.number_input(
                    feature.replace('_', ' ').title(),
                    value=float(feature_data.median()),
                    min_value=float(feature_data.min()),
                    max_value=float(feature_data.max()),
                    key=f"comp_{feature}"
                )
        
        # Fill remaining features with median values
        all_features = [col for col in st.session_state.engineered_data.columns if col != 'target']
        for feature in all_features:
            if feature not in comparison_data:
                if feature in numerical_features:
                    comparison_data[feature] = st.session_state.engineered_data[feature].median()
                else:
                    comparison_data[feature] = st.session_state.engineered_data[feature].mode()[0]
        
        if st.button("🔄 Compare Models", type="primary"):
            try:
                # Prepare input
                input_df = pd.DataFrame([comparison_data])
                input_df = input_df[all_features]
                
                # Get predictions from all models
                model_predictions = {}
                
                for model_name, model in st.session_state.trained_models.items():
                    prediction = model.predict(input_df)[0]
                    
                    if hasattr(model, 'predict_proba'):
                        probability = model.predict_proba(input_df)[0, 1]
                    else:
                        probability = float(prediction)
                    
                    credit_score = int(300 + (probability * 550))
                    decision = "APPROVED" if probability >= threshold else "REJECTED"
                    
                    model_predictions[model_name] = {
                        'Prediction': prediction,
                        'Probability': probability,
                        'Credit_Score': credit_score,
                        'Decision': decision
                    }
                
                # Display comparison results
                st.subheader("📊 Model Comparison Results")
                
                # Create comparison dataframe
                comparison_df = pd.DataFrame(model_predictions).T
                comparison_df.index.name = 'Model'
                
                # Display table
                st.dataframe(comparison_df, use_container_width=True)
                
                # Visualization
                col1, col2 = st.columns(2)
                
                with col1:
                    # Probability comparison
                    fig = px.bar(
                        x=list(model_predictions.keys()),
                        y=[v['Probability'] for v in model_predictions.values()],
                        title="Default Probability by Model"
                    )
                    fig.add_hline(y=threshold, line_dash="dash", 
                                annotation_text="Approval Threshold")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Credit score comparison
                    fig = px.bar(
                        x=list(model_predictions.keys()),
                        y=[v['Credit_Score'] for v in model_predictions.values()],
                        title="Credit Score by Model"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Decision consensus
                decisions = [v['Decision'] for v in model_predictions.values()]
                approved_count = decisions.count('APPROVED')
                consensus = approved_count / len(decisions)
                
                if consensus >= 0.8:
                    st.success(f"✅ **Strong Consensus**: {approved_count}/{len(decisions)} models recommend APPROVAL")
                elif consensus >= 0.6:
                    st.warning(f"⚠️ **Moderate Consensus**: {approved_count}/{len(decisions)} models recommend approval")
                else:
                    st.error(f"❌ **Low Consensus**: Only {approved_count}/{len(decisions)} models recommend approval")
                
                # Model agreement analysis
                st.subheader("🤝 Model Agreement Analysis")
                
                probabilities = [v['Probability'] for v in model_predictions.values()]
                prob_std = np.std(probabilities)
                prob_range = max(probabilities) - min(probabilities)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Probability Range", f"{prob_range:.3f}")
                with col2:
                    st.metric("Standard Deviation", f"{prob_std:.3f}")
                with col3:
                    agreement_level = "High" if prob_std < 0.1 else "Medium" if prob_std < 0.2 else "Low"
                    st.metric("Agreement Level", agreement_level)
                
            except Exception as e:
                st.error(f"❌ Error in model comparison: {str(e)}")
    
    # Prediction history
    if 'prediction_history' in st.session_state and st.session_state.prediction_history:
        st.markdown("---")
        st.subheader("📜 Prediction History")
        
        history_df = pd.DataFrame(st.session_state.prediction_history)
        
        # Summary statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Predictions", len(history_df))
        with col2:
            approved_in_history = sum(history_df['decision'] == 'APPROVED')
            st.metric("Approved", approved_in_history)
        with col3:
            avg_score_history = history_df['credit_score'].mean()
            st.metric("Average Score", f"{avg_score_history:.0f}")
        
        # Display recent predictions
        st.dataframe(
            history_df[['timestamp', 'model_used', 'probability', 'credit_score', 'decision']].tail(10),
            use_container_width=True
        )
        
        # Download history
        if st.button("📥 Download Prediction History"):
            csv = history_df.to_csv(index=False)
            st.download_button(
                label="Download History",
                data=csv,
                file_name=f"prediction_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()
