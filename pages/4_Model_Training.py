import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from utils.model_trainer import ModelTrainer
from utils.bias_detector import BiasDetector

st.set_page_config(page_title="Model Training", page_icon="🤖", layout="wide")

def main():
    st.title("🤖 Model Training & Hyperparameter Tuning")
    
    if st.session_state.engineered_data is None:
        st.error("❌ No engineered data found. Please complete feature engineering first.")
        st.info("👈 Go to Feature Engineering page to prepare your features.")
        return
    
    data = st.session_state.engineered_data.copy()
    trainer = ModelTrainer()
    bias_detector = BiasDetector()
    
    # Check for target variable
    if 'target' not in data.columns:
        st.error("❌ Target variable 'target' not found in the dataset.")
        return
    
    st.markdown("Train and tune multiple classification models for credit scoring.")
    
    # Model configuration
    st.subheader("🔧 Training Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Data Split:**")
        test_size = st.slider("Test set size", 0.1, 0.4, 0.2, 0.05)
        random_state = st.number_input("Random state", value=42, min_value=0)
        stratify_split = st.checkbox("Stratified split", value=True, 
                                   help="Maintain class distribution in train/test sets")
        
        st.markdown("**Cross-Validation:**")
        cv_folds = st.slider("CV folds", 3, 10, 5)
        cv_scoring = st.selectbox("CV scoring metric", 
                                ['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
                                index=4)
    
    with col2:
        st.markdown("**Model Selection:**")
        models_to_train = st.multiselect(
            "Select models to train",
            options=['Logistic Regression', 'Decision Tree', 'Random Forest', 
                    'Gradient Boosting', 'SVM'],
            default=['Logistic Regression', 'Decision Tree', 'Random Forest'],
            help="Choose which models to train and compare"
        )
        
        perform_tuning = st.checkbox("Hyperparameter tuning", value=True,
                                   help="Perform grid search for optimal parameters")
        
        class_weight = st.selectbox("Class weighting", 
                                  ['balanced', 'none'], 
                                  index=0,
                                  help="Handle class imbalance")
    
    # Advanced options
    with st.expander("🔧 Advanced Options"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Feature Selection:**")
            feature_selection = st.checkbox("Recursive feature elimination", value=False)
            if feature_selection:
                n_features_to_select = st.slider("Number of features", 5, 20, 10)
        
        with col2:
            st.markdown("**Bias Detection:**")
            check_bias = st.checkbox("Check for bias", value=True)
            if check_bias:
                protected_attributes = st.multiselect(
                    "Protected attributes",
                    options=[col for col in data.columns if col != 'target'],
                    help="Select features to check for discriminatory bias"
                )
    
    # Hyperparameter grids
    if perform_tuning:
        st.subheader("⚙️ Hyperparameter Grids")
        
        param_tabs = st.tabs(models_to_train)
        param_grids = {}
        
        for i, model_name in enumerate(models_to_train):
            with param_tabs[i]:
                if model_name == 'Logistic Regression':
                    col1, col2 = st.columns(2)
                    with col1:
                        c_values = st.text_input("C values (comma-separated)", 
                                               value="0.1, 1, 10")
                        solver_values = st.multiselect("Solvers", 
                                                     ['liblinear', 'lbfgs', 'saga'],
                                                     default=['liblinear', 'lbfgs'])
                    with col2:
                        max_iter_values = st.text_input("Max iterations", value="1000")
                    
                    param_grids['Logistic Regression'] = {
                        'C': [float(x.strip()) for x in c_values.split(',')],
                        'solver': solver_values,
                        'max_iter': [int(max_iter_values)]
                    }
                
                elif model_name == 'Decision Tree':
                    col1, col2 = st.columns(2)
                    with col1:
                        max_depth_values = st.text_input("Max depth (comma-separated)", 
                                                       value="3, 5, 10, None")
                        min_samples_split = st.text_input("Min samples split", 
                                                        value="2, 5, 10")
                    with col2:
                        min_samples_leaf = st.text_input("Min samples leaf", 
                                                       value="1, 2, 4")
                        criterion_values = st.multiselect("Criterion", 
                                                        ['gini', 'entropy'],
                                                        default=['gini'])
                    
                    max_depth_list = []
                    for val in max_depth_values.split(','):
                        val = val.strip()
                        if val.lower() == 'none':
                            max_depth_list.append(None)
                        else:
                            max_depth_list.append(int(val))
                    
                    param_grids['Decision Tree'] = {
                        'max_depth': max_depth_list,
                        'min_samples_split': [int(x.strip()) for x in min_samples_split.split(',')],
                        'min_samples_leaf': [int(x.strip()) for x in min_samples_leaf.split(',')],
                        'criterion': criterion_values
                    }
                
                elif model_name == 'Random Forest':
                    col1, col2 = st.columns(2)
                    with col1:
                        n_estimators_values = st.text_input("N estimators", 
                                                          value="50, 100, 200")
                        max_depth_values = st.text_input("Max depth", 
                                                       value="3, 5, 10, None")
                    with col2:
                        min_samples_split = st.text_input("Min samples split", 
                                                        value="2, 5")
                        max_features = st.multiselect("Max features", 
                                                    ['sqrt', 'log2', 'auto'],
                                                    default=['sqrt'])
                    
                    max_depth_list = []
                    for val in max_depth_values.split(','):
                        val = val.strip()
                        if val.lower() == 'none':
                            max_depth_list.append(None)
                        else:
                            max_depth_list.append(int(val))
                    
                    param_grids['Random Forest'] = {
                        'n_estimators': [int(x.strip()) for x in n_estimators_values.split(',')],
                        'max_depth': max_depth_list,
                        'min_samples_split': [int(x.strip()) for x in min_samples_split.split(',')],
                        'max_features': max_features
                    }
    
    # Train models button
    if st.button("🚀 Train Models", type="primary"):
        if not models_to_train:
            st.error("Please select at least one model to train.")
            return
        
        with st.spinner("Training models... This may take a few minutes."):
            try:
                # Prepare data
                X = data.drop('target', axis=1)
                y = data['target']
                
                # Configure training
                config = {
                    'test_size': test_size,
                    'random_state': random_state,
                    'stratify_split': stratify_split,
                    'cv_folds': cv_folds,
                    'cv_scoring': cv_scoring,
                    'models_to_train': models_to_train,
                    'perform_tuning': perform_tuning,
                    'class_weight': class_weight if class_weight != 'none' else None,
                    'param_grids': param_grids if perform_tuning else {}
                }
                
                # Train models
                results = trainer.train_models(X, y, config)
                
                # Store results
                st.session_state.trained_models = results['models']
                st.session_state.model_results = results['results']
                st.session_state.X_train = results['X_train']
                st.session_state.X_test = results['X_test']
                st.session_state.y_train = results['y_train']
                st.session_state.y_test = results['y_test']
                
                # Bias detection
                if check_bias and protected_attributes:
                    bias_results = {}
                    for model_name, model in results['models'].items():
                        try:
                            y_pred = model.predict(results['X_test'])
                            bias_metrics = bias_detector.detect_bias(
                                results['X_test'], results['y_test'], y_pred, 
                                protected_attributes
                            )
                            bias_results[model_name] = bias_metrics
                        except Exception as e:
                            st.warning(f"Bias detection failed for {model_name}: {str(e)}")
                    
                    st.session_state.bias_results = bias_results
                
                st.success("✅ Model training completed!")
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error during training: {str(e)}")
    
    # Show results if models have been trained
    if st.session_state.trained_models:
        st.markdown("---")
        st.subheader("📊 Training Results")
        
        results = st.session_state.model_results
        
        # Model comparison table
        st.markdown("### 🏆 Model Performance Comparison")
        
        comparison_data = []
        for model_name, metrics in results.items():
            comparison_data.append({
                'Model': model_name,
                'CV Score': f"{metrics['cv_score_mean']:.3f} (±{metrics['cv_score_std']:.3f})",
                'Test Accuracy': f"{metrics['test_accuracy']:.3f}",
                'Test Precision': f"{metrics['test_precision']:.3f}",
                'Test Recall': f"{metrics['test_recall']:.3f}",
                'Test F1': f"{metrics['test_f1']:.3f}",
                'Test ROC-AUC': f"{metrics['test_roc_auc']:.3f}",
                'Training Time': f"{metrics['training_time']:.2f}s"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
        
        # Best model identification
        best_model_name = max(results.keys(), 
                            key=lambda x: results[x]['test_roc_auc'])
        st.session_state.best_model = best_model_name
        
        st.success(f"🥇 Best performing model: **{best_model_name}** (ROC-AUC: {results[best_model_name]['test_roc_auc']:.3f})")
        
        # Detailed results for each model
        st.markdown("### 📈 Detailed Model Analysis")
        
        selected_model = st.selectbox(
            "Select model for detailed analysis",
            options=list(results.keys()),
            index=list(results.keys()).index(best_model_name) if best_model_name in results else 0
        )
        
        if selected_model:
            model_metrics = results[selected_model]
            
            # Metrics overview
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Cross-Validation Score", 
                         f"{model_metrics['cv_score_mean']:.3f}")
            with col2:
                st.metric("Test Accuracy", 
                         f"{model_metrics['test_accuracy']:.3f}")
            with col3:
                st.metric("Test F1-Score", 
                         f"{model_metrics['test_f1']:.3f}")
            with col4:
                st.metric("Test ROC-AUC", 
                         f"{model_metrics['test_roc_auc']:.3f}")
            
            # Confusion matrix
            col1, col2 = st.columns(2)
            
            with col1:
                if 'confusion_matrix' in model_metrics:
                    cm = model_metrics['confusion_matrix']
                    fig = px.imshow(cm, 
                                  text_auto=True,
                                  aspect="auto",
                                  title=f"Confusion Matrix - {selected_model}",
                                  labels=dict(x="Predicted", y="Actual"))
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if 'feature_importance' in model_metrics:
                    importance = model_metrics['feature_importance']
                    top_features = dict(sorted(importance.items(), 
                                             key=lambda x: x[1], reverse=True)[:10])
                    
                    fig = px.bar(
                        x=list(top_features.values()),
                        y=list(top_features.keys()),
                        orientation='h',
                        title=f"Top 10 Feature Importances - {selected_model}"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Classification report
            if 'classification_report' in model_metrics:
                st.markdown("**Classification Report:**")
                st.text(model_metrics['classification_report'])
            
            # Hyperparameters
            if 'best_params' in model_metrics:
                st.markdown("**Best Hyperparameters:**")
                for param, value in model_metrics['best_params'].items():
                    st.write(f"- {param}: {value}")
        
        # Cross-validation scores visualization
        st.markdown("### 📊 Cross-Validation Performance")
        
        cv_data = []
        for model_name, metrics in results.items():
            cv_data.append({
                'Model': model_name,
                'CV_Score': metrics['cv_score_mean'],
                'CV_Std': metrics['cv_score_std']
            })
        
        cv_df = pd.DataFrame(cv_data)
        fig = px.bar(cv_df, x='Model', y='CV_Score', error_y='CV_Std',
                    title="Cross-Validation Scores with Standard Deviation")
        st.plotly_chart(fig, use_container_width=True)
        
        # Model performance radar chart
        st.markdown("### 🕸️ Model Performance Radar Chart")
        
        metrics_to_plot = ['test_accuracy', 'test_precision', 'test_recall', 
                          'test_f1', 'test_roc_auc']
        
        fig = go.Figure()
        
        for model_name, metrics in results.items():
            values = [metrics[metric] for metric in metrics_to_plot]
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
                fill='toself',
                name=model_name
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Model Performance Comparison"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Bias analysis results
        if hasattr(st.session_state, 'bias_results') and st.session_state.bias_results:
            st.markdown("### ⚖️ Bias Analysis Results")
            
            for model_name, bias_metrics in st.session_state.bias_results.items():
                with st.expander(f"Bias Analysis - {model_name}"):
                    for attribute, metrics in bias_metrics.items():
                        st.write(f"**{attribute}:**")
                        for metric_name, value in metrics.items():
                            st.write(f"- {metric_name}: {value:.3f}")
        
        # Next steps
        st.subheader("➡️ Next Steps")
        st.info("""
        Model training completed! You can now proceed to:
        1. **Model Evaluation** - Detailed analysis and interpretation of model performance
        2. **Credit Scoring** - Use the trained models for making predictions
        """)
    
    else:
        # Show data readiness status
        st.subheader("📊 Data Readiness Check")
        
        X = data.drop('target', axis=1)
        y = data['target']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Samples", len(data))
        with col2:
            st.metric("Features", len(X.columns))
        with col3:
            class_counts = y.value_counts()
            balance_ratio = min(class_counts) / max(class_counts)
            st.metric("Class Balance", f"{balance_ratio:.2f}")
        with col4:
            missing_count = X.isnull().sum().sum()
            st.metric("Missing Values", missing_count)
        
        # Data quality checks
        st.markdown("### ✅ Pre-training Checklist")
        
        checks = []
        
        # Check for sufficient data
        if len(data) >= 100:
            checks.append("✅ Sufficient data samples (≥100)")
        else:
            checks.append("❌ Insufficient data samples (<100)")
        
        # Check for missing values
        if X.isnull().sum().sum() == 0:
            checks.append("✅ No missing values")
        else:
            checks.append("⚠️ Missing values detected")
        
        # Check class balance
        if balance_ratio >= 0.3:
            checks.append("✅ Reasonable class balance")
        else:
            checks.append("⚠️ Class imbalance detected")
        
        # Check for categorical columns
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) == 0:
            checks.append("✅ All features are numerical")
        else:
            checks.append(f"⚠️ {len(categorical_cols)} categorical columns need encoding")
        
        for check in checks:
            st.write(check)

if __name__ == "__main__":
    main()
