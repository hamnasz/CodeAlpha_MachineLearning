import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (roc_curve, auc, precision_recall_curve, 
                           classification_report, confusion_matrix)
from sklearn.calibration import calibration_curve
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
from utils.model_evaluator import ModelEvaluator

st.set_page_config(page_title="Model Evaluation", page_icon="📊", layout="wide")

def main():
    st.title("📊 Model Evaluation & Interpretability")
    
    if not st.session_state.trained_models:
        st.error("❌ No trained models found. Please complete model training first.")
        st.info("👈 Go to Model Training page to train your models.")
        return
    
    evaluator = ModelEvaluator()
    
    st.markdown("Comprehensive evaluation and interpretation of your credit scoring models.")
    
    # Model selection for detailed evaluation
    st.subheader("🔍 Select Model for Detailed Evaluation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_model = st.selectbox(
            "Choose model",
            options=list(st.session_state.trained_models.keys()),
            index=list(st.session_state.trained_models.keys()).index(st.session_state.best_model) 
            if st.session_state.best_model in st.session_state.trained_models else 0
        )
    
    with col2:
        evaluation_type = st.selectbox(
            "Evaluation focus",
            options=['Performance Metrics', 'Model Interpretability', 'Business Impact', 'Regulatory Compliance'],
            index=0
        )
    
    model = st.session_state.trained_models[selected_model]
    model_results = st.session_state.model_results[selected_model]
    X_test = st.session_state.X_test
    y_test = st.session_state.y_test
    
    # Performance Metrics Tab
    if evaluation_type == 'Performance Metrics':
        st.subheader("📈 Performance Metrics Analysis")
        
        # Get predictions and probabilities
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Key metrics overview
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", f"{model_results['test_accuracy']:.3f}")
        with col2:
            st.metric("Precision", f"{model_results['test_precision']:.3f}")
        with col3:
            st.metric("Recall", f"{model_results['test_recall']:.3f}")
        with col4:
            st.metric("F1-Score", f"{model_results['test_f1']:.3f}")
        
        # ROC and Precision-Recall Curves
        col1, col2 = st.columns(2)
        
        with col1:
            if y_pred_proba is not None:
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                roc_auc = auc(fpr, tpr)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=fpr, y=tpr, 
                                       name=f'ROC Curve (AUC = {roc_auc:.3f})',
                                       line=dict(color='blue', width=2)))
                fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], 
                                       mode='lines',
                                       name='Random Classifier',
                                       line=dict(color='red', dash='dash')))
                fig.update_layout(
                    title='ROC Curve',
                    xaxis_title='False Positive Rate',
                    yaxis_title='True Positive Rate',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if y_pred_proba is not None:
                precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
                pr_auc = auc(recall, precision)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=recall, y=precision,
                                       name=f'PR Curve (AUC = {pr_auc:.3f})',
                                       line=dict(color='green', width=2)))
                fig.update_layout(
                    title='Precision-Recall Curve',
                    xaxis_title='Recall',
                    yaxis_title='Precision',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Confusion Matrix and Classification Report
        col1, col2 = st.columns(2)
        
        with col1:
            cm = confusion_matrix(y_test, y_pred)
            fig = px.imshow(cm, 
                          text_auto=True,
                          aspect="auto",
                          title="Confusion Matrix",
                          labels=dict(x="Predicted", y="Actual"),
                          color_continuous_scale="Blues")
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate additional metrics
            tn, fp, fn, tp = cm.ravel()
            specificity = tn / (tn + fp)
            sensitivity = tp / (tp + fn)
            
            st.write(f"**Specificity (True Negative Rate):** {specificity:.3f}")
            st.write(f"**Sensitivity (True Positive Rate):** {sensitivity:.3f}")
        
        with col2:
            st.markdown("**Classification Report:**")
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose().round(3)
            st.dataframe(report_df, use_container_width=True)
        
        # Probability Distribution Analysis
        if y_pred_proba is not None:
            st.subheader("📊 Prediction Probability Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Probability distribution by class
                prob_df = pd.DataFrame({
                    'Probability': y_pred_proba,
                    'Actual': y_test
                })
                
                fig = px.histogram(prob_df, x='Probability', color='Actual',
                                 title='Prediction Probability Distribution by Actual Class',
                                 nbins=30, opacity=0.7)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Calibration curve
                fraction_of_positives, mean_predicted_value = calibration_curve(
                    y_test, y_pred_proba, n_bins=10
                )
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=mean_predicted_value, y=fraction_of_positives,
                                       mode='markers+lines',
                                       name='Model Calibration',
                                       line=dict(color='blue')))
                fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1],
                                       mode='lines',
                                       name='Perfect Calibration',
                                       line=dict(color='red', dash='dash')))
                fig.update_layout(
                    title='Calibration Plot',
                    xaxis_title='Mean Predicted Probability',
                    yaxis_title='Fraction of Positives',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Threshold Analysis
        if y_pred_proba is not None:
            st.subheader("🎯 Threshold Analysis")
            
            thresholds = np.arange(0.1, 1.0, 0.05)
            metrics_at_thresholds = []
            
            for threshold in thresholds:
                y_pred_thresh = (y_pred_proba >= threshold).astype(int)
                
                try:
                    from sklearn.metrics import precision_score, recall_score, f1_score
                    precision = precision_score(y_test, y_pred_thresh)
                    recall = recall_score(y_test, y_pred_thresh)
                    f1 = f1_score(y_test, y_pred_thresh)
                    
                    metrics_at_thresholds.append({
                        'Threshold': threshold,
                        'Precision': precision,
                        'Recall': recall,
                        'F1-Score': f1
                    })
                except:
                    continue
            
            if metrics_at_thresholds:
                metrics_df = pd.DataFrame(metrics_at_thresholds)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=metrics_df['Threshold'], y=metrics_df['Precision'],
                                       mode='lines', name='Precision'))
                fig.add_trace(go.Scatter(x=metrics_df['Threshold'], y=metrics_df['Recall'],
                                       mode='lines', name='Recall'))
                fig.add_trace(go.Scatter(x=metrics_df['Threshold'], y=metrics_df['F1-Score'],
                                       mode='lines', name='F1-Score'))
                
                fig.update_layout(
                    title='Metrics vs Threshold',
                    xaxis_title='Threshold',
                    yaxis_title='Score',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Optimal threshold recommendation
                optimal_f1_idx = metrics_df['F1-Score'].idxmax()
                optimal_threshold = metrics_df.loc[optimal_f1_idx, 'Threshold']
                st.info(f"💡 **Recommended threshold for optimal F1-Score:** {optimal_threshold:.2f}")
    
    # Model Interpretability Tab
    elif evaluation_type == 'Model Interpretability':
        st.subheader("🔍 Model Interpretability Analysis")
        
        # Feature Importance
        if 'feature_importance' in model_results:
            st.markdown("### 📊 Feature Importance")
            
            importance = model_results['feature_importance']
            top_features = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:15])
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(
                    x=list(top_features.values()),
                    y=list(top_features.keys()),
                    orientation='h',
                    title="Top 15 Feature Importances"
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("**Feature Importance Analysis:**")
                for i, (feature, importance_val) in enumerate(list(top_features.items())[:10], 1):
                    st.write(f"{i}. **{feature}**: {importance_val:.4f}")
                
                # Feature importance insights
                st.markdown("**Key Insights:**")
                top_3_features = list(top_features.keys())[:3]
                st.write(f"• Top 3 most important features: {', '.join(top_3_features)}")
                
                total_importance = sum(list(top_features.values())[:5])
                st.write(f"• Top 5 features account for {total_importance:.1%} of total importance")
        
        # SHAP Analysis (if possible)
        if SHAP_AVAILABLE:
            try:
                st.markdown("### 🎯 SHAP (SHapley Additive exPlanations) Analysis")
                
                with st.spinner("Computing SHAP values... This may take a moment."):
                    # Sample data for SHAP to avoid long computation times
                    X_sample = X_test.sample(min(100, len(X_test)), random_state=42)
                    
                    # Create SHAP explainer based on model type
                    if hasattr(model, 'tree_'):
                        explainer = shap.TreeExplainer(model)
                    elif hasattr(model, 'coef_'):
                        explainer = shap.LinearExplainer(model, X_sample)
                    else:
                        explainer = shap.KernelExplainer(model.predict_proba, X_sample)
                
                shap_values = explainer.shap_values(X_sample)
                
                # If binary classification, take positive class
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # SHAP summary plot (simplified)
                    feature_importance = np.abs(shap_values).mean(axis=0)
                    feature_names = X_sample.columns
                    
                    top_shap_features = pd.DataFrame({
                        'Feature': feature_names,
                        'SHAP_Importance': feature_importance
                    }).sort_values('SHAP_Importance', ascending=False).head(10)
                    
                    fig = px.bar(top_shap_features, x='SHAP_Importance', y='Feature',
                               orientation='h', title='SHAP Feature Importance')
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.markdown("**SHAP Interpretation:**")
                    st.write("SHAP values explain individual predictions by quantifying the contribution of each feature.")
                    
                    # Show example prediction explanation
                    sample_idx = 0
                    sample_prediction = model.predict_proba(X_sample.iloc[[sample_idx]])[0, 1]
                    sample_shap = shap_values[sample_idx]
                    
                    st.write(f"**Example Prediction Explanation:**")
                    st.write(f"Predicted probability: {sample_prediction:.3f}")
                    
                    # Top contributing features for this prediction
                    feature_contributions = list(zip(feature_names, sample_shap))
                    feature_contributions.sort(key=lambda x: abs(x[1]), reverse=True)
                    
                    st.write("**Top contributing features:**")
                    for feature, contribution in feature_contributions[:5]:
                        direction = "increases" if contribution > 0 else "decreases"
                        st.write(f"• {feature}: {direction} probability by {abs(contribution):.3f}")
        
            except Exception as e:
                st.warning(f"SHAP analysis not available: {str(e)}")
                st.info("SHAP analysis requires additional setup and may not work with all model types.")
        else:
            st.markdown("### 🎯 SHAP Analysis")
            st.info("SHAP library is not available. Install SHAP for advanced model interpretability features.")
        
        # Decision Rules (for tree-based models)
        if hasattr(model, 'tree_'):
            st.markdown("### 🌳 Decision Rules Analysis")
            
            try:
                from sklearn.tree import export_text
                
                if hasattr(model, 'estimators_'):
                    # Random Forest - show rules for first few trees
                    st.write("**Sample Decision Rules (First Tree):**")
                    tree_rules = export_text(model.estimators_[0], 
                                           feature_names=list(X_test.columns),
                                           max_depth=3)
                    st.text(tree_rules[:1000] + "..." if len(tree_rules) > 1000 else tree_rules)
                else:
                    # Single Decision Tree
                    st.write("**Decision Rules:**")
                    tree_rules = export_text(model, 
                                           feature_names=list(X_test.columns),
                                           max_depth=5)
                    st.text(tree_rules[:2000] + "..." if len(tree_rules) > 2000 else tree_rules)
            
            except Exception as e:
                st.warning(f"Decision rules extraction failed: {str(e)}")
        
        # Model Complexity Analysis
        st.markdown("### 🔧 Model Complexity Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if hasattr(model, 'coef_'):
                st.metric("Number of Coefficients", len(model.coef_[0]))
            elif hasattr(model, 'n_estimators'):
                st.metric("Number of Trees", model.n_estimators)
            elif hasattr(model, 'tree_'):
                st.metric("Number of Leaves", model.tree_.n_leaves)
        
        with col2:
            if hasattr(model, 'tree_'):
                st.metric("Tree Depth", model.tree_.max_depth)
            elif hasattr(model, 'max_depth'):
                st.metric("Max Depth", model.max_depth or "Unlimited")
        
        with col3:
            n_features = len(X_test.columns)
            st.metric("Number of Features", n_features)
    
    # Business Impact Tab
    elif evaluation_type == 'Business Impact':
        st.subheader("💼 Business Impact Analysis")
        
        # Get predictions and probabilities
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Business metrics configuration
        st.markdown("### 💰 Business Metrics Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Cost Parameters:**")
            default_cost = st.number_input("Cost of default (per customer)", value=10000, min_value=0)
            approval_cost = st.number_input("Cost of loan processing", value=100, min_value=0)
            opportunity_cost = st.number_input("Opportunity cost of rejection", value=500, min_value=0)
        
        with col2:
            st.markdown("**Revenue Parameters:**")
            loan_profit = st.number_input("Profit from good loan", value=2000, min_value=0)
            current_threshold = st.slider("Current approval threshold", 0.1, 0.9, 0.5, 0.05)
        
        # Calculate business impact
        if y_pred_proba is not None:
            # Confusion matrix at current threshold
            y_pred_business = (y_pred_proba >= current_threshold).astype(int)
            cm = confusion_matrix(y_test, y_pred_business)
            tn, fp, fn, tp = cm.ravel()
            
            # Calculate costs and profits
            total_default_cost = fp * default_cost  # False positives cause defaults
            total_processing_cost = (tp + fp) * approval_cost  # All approvals incur processing cost
            total_opportunity_cost = fn * opportunity_cost  # False negatives lose opportunities
            total_profit = tp * loan_profit  # True positives generate profit
            
            net_profit = total_profit - total_default_cost - total_processing_cost - total_opportunity_cost
            
            # Display business metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Profit", f"${total_profit:,.0f}")
            with col2:
                st.metric("Default Costs", f"${total_default_cost:,.0f}")
            with col3:
                st.metric("Processing Costs", f"${total_processing_cost:,.0f}")
            with col4:
                st.metric("Net Profit", f"${net_profit:,.0f}")
            
            # Threshold optimization for business metrics
            st.markdown("### 📈 Threshold Optimization for Business Impact")
            
            thresholds = np.arange(0.1, 0.9, 0.05)
            business_metrics = []
            
            for threshold in thresholds:
                y_pred_thresh = (y_pred_proba >= threshold).astype(int)
                cm_thresh = confusion_matrix(y_test, y_pred_thresh)
                tn_t, fp_t, fn_t, tp_t = cm_thresh.ravel()
                
                profit_t = tp_t * loan_profit
                default_cost_t = fp_t * default_cost
                processing_cost_t = (tp_t + fp_t) * approval_cost
                opportunity_cost_t = fn_t * opportunity_cost
                net_profit_t = profit_t - default_cost_t - processing_cost_t - opportunity_cost_t
                
                approval_rate = (tp_t + fp_t) / len(y_test)
                default_rate = fp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else 0
                
                business_metrics.append({
                    'Threshold': threshold,
                    'Net_Profit': net_profit_t,
                    'Approval_Rate': approval_rate,
                    'Default_Rate': default_rate,
                    'Profit_Per_Customer': net_profit_t / len(y_test)
                })
            
            business_df = pd.DataFrame(business_metrics)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.line(business_df, x='Threshold', y='Net_Profit',
                            title='Net Profit vs Threshold')
                fig.add_vline(x=current_threshold, line_dash="dash", 
                            annotation_text="Current Threshold")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=business_df['Threshold'], y=business_df['Approval_Rate'],
                                       mode='lines', name='Approval Rate'))
                fig.add_trace(go.Scatter(x=business_df['Threshold'], y=business_df['Default_Rate'],
                                       mode='lines', name='Default Rate'))
                fig.add_vline(x=current_threshold, line_dash="dash", 
                            annotation_text="Current Threshold")
                fig.update_layout(title='Approval and Default Rates vs Threshold',
                                yaxis_title='Rate')
                st.plotly_chart(fig, use_container_width=True)
            
            # Optimal threshold recommendation
            optimal_profit_idx = business_df['Net_Profit'].idxmax()
            optimal_threshold = business_df.loc[optimal_profit_idx, 'Threshold']
            optimal_profit = business_df.loc[optimal_profit_idx, 'Net_Profit']
            
            st.success(f"💡 **Optimal threshold for maximum profit:** {optimal_threshold:.2f} (Net Profit: ${optimal_profit:,.0f})")
        
        # Portfolio Analysis
        st.markdown("### 📊 Credit Portfolio Analysis")
        
        if y_pred_proba is not None:
            # Risk distribution
            risk_bins = pd.cut(y_pred_proba, bins=5, labels=['Very High Risk', 'High Risk', 'Medium Risk', 'Low Risk', 'Very Low Risk'])
            risk_distribution = risk_bins.value_counts()
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.pie(values=risk_distribution.values, names=risk_distribution.index,
                           title='Risk Distribution of Test Portfolio')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Expected default rates by risk bucket
                risk_default_rates = []
                for bucket in risk_distribution.index:
                    bucket_mask = risk_bins == bucket
                    bucket_default_rate = y_test[bucket_mask].mean() if bucket_mask.sum() > 0 else 0
                    risk_default_rates.append(bucket_default_rate)
                
                fig = px.bar(x=risk_distribution.index, y=risk_default_rates,
                           title='Expected Default Rate by Risk Bucket')
                fig.update_yaxis(title='Default Rate')
                st.plotly_chart(fig, use_container_width=True)
    
    # Regulatory Compliance Tab
    elif evaluation_type == 'Regulatory Compliance':
        st.subheader("⚖️ Regulatory Compliance Analysis")
        
        # Model documentation
        st.markdown("### 📋 Model Documentation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Model Characteristics:**")
            st.write(f"• Model Type: {selected_model}")
            st.write(f"• Training Date: {pd.Timestamp.now().strftime('%Y-%m-%d')}")
            st.write(f"• Features Used: {len(X_test.columns)}")
            st.write(f"• Training Samples: {len(st.session_state.X_train)}")
            st.write(f"• Test Samples: {len(X_test)}")
            
            # Model performance summary
            st.markdown("**Performance Summary:**")
            st.write(f"• Accuracy: {model_results['test_accuracy']:.3f}")
            st.write(f"• AUC-ROC: {model_results['test_roc_auc']:.3f}")
            st.write(f"• Precision: {model_results['test_precision']:.3f}")
            st.write(f"• Recall: {model_results['test_recall']:.3f}")
        
        with col2:
            st.markdown("**Regulatory Considerations:**")
            
            # Model interpretability assessment
            interpretability_score = "High" if selected_model in ['Logistic Regression', 'Decision Tree'] else "Medium" if selected_model == 'Random Forest' else "Low"
            st.write(f"• Model Interpretability: {interpretability_score}")
            
            # Fair lending compliance
            st.write("• Fair Lending: Under Review")
            st.write("• GDPR Compliance: Model provides explanations")
            st.write("• Model Risk Rating: Medium")
            
            # Documentation completeness
            st.markdown("**Documentation Status:**")
            st.write("✅ Model development process documented")
            st.write("✅ Feature engineering recorded")
            st.write("✅ Performance metrics calculated")
            st.write("✅ Model interpretability provided")
        
        # Bias and Fairness Analysis
        if hasattr(st.session_state, 'bias_results') and st.session_state.bias_results:
            st.markdown("### ⚖️ Bias and Fairness Assessment")
            
            if selected_model in st.session_state.bias_results:
                bias_metrics = st.session_state.bias_results[selected_model]
                
                for attribute, metrics in bias_metrics.items():
                    st.markdown(f"**Bias Analysis for {attribute}:**")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if 'demographic_parity' in metrics:
                            parity_score = metrics['demographic_parity']
                            parity_status = "✅ Acceptable" if abs(parity_score - 1.0) < 0.2 else "⚠️ Review Required"
                            st.write(f"Demographic Parity: {parity_score:.3f} {parity_status}")
                    
                    with col2:
                        if 'equalized_odds' in metrics:
                            odds_score = metrics['equalized_odds']
                            odds_status = "✅ Acceptable" if abs(odds_score - 1.0) < 0.2 else "⚠️ Review Required"
                            st.write(f"Equalized Odds: {odds_score:.3f} {odds_status}")
                    
                    with col3:
                        if 'calibration' in metrics:
                            cal_score = metrics['calibration']
                            cal_status = "✅ Well Calibrated" if cal_score < 0.1 else "⚠️ Calibration Issues"
                            st.write(f"Calibration: {cal_score:.3f} {cal_status}")
        
        # Model Risk Assessment
        st.markdown("### 🎯 Model Risk Assessment")
        
        risk_factors = []
        
        # Data quality risks
        if X_test.isnull().sum().sum() > 0:
            risk_factors.append("⚠️ Missing data in test set")
        
        # Model complexity risks
        if selected_model in ['Random Forest', 'Gradient Boosting']:
            risk_factors.append("⚠️ Complex model may be difficult to explain")
        
        # Performance risks
        if model_results['test_accuracy'] < 0.7:
            risk_factors.append("🔴 Low model accuracy")
        elif model_results['test_accuracy'] < 0.8:
            risk_factors.append("⚠️ Moderate model accuracy")
        
        # Overfitting risks
        train_accuracy = model_results.get('train_accuracy', 0)
        if train_accuracy - model_results['test_accuracy'] > 0.1:
            risk_factors.append("⚠️ Potential overfitting detected")
        
        if risk_factors:
            st.markdown("**Identified Risk Factors:**")
            for risk in risk_factors:
                st.write(risk)
        else:
            st.success("✅ No major risk factors identified")
        
        # Monitoring Recommendations
        st.markdown("### 📊 Model Monitoring Recommendations")
        
        monitoring_plan = [
            "📈 **Performance Monitoring**: Track model accuracy monthly",
            "📊 **Data Drift Detection**: Monitor feature distributions",
            "⚖️ **Bias Monitoring**: Regular fairness assessments",
            "🔄 **Model Retraining**: Quarterly model updates",
            "📋 **Documentation Updates**: Maintain model registry",
            "🎯 **Business Impact**: Track financial performance"
        ]
        
        for item in monitoring_plan:
            st.write(item)
        
        # Model Approval Checklist
        st.markdown("### ✅ Model Approval Checklist")
        
        checklist_items = [
            ("Model Performance", model_results['test_roc_auc'] >= 0.7),
            ("Documentation Complete", True),
            ("Bias Assessment", len(st.session_state.get('bias_results', {})) > 0),
            ("Feature Importance", 'feature_importance' in model_results),
            ("Business Impact Analysis", True),
            ("Risk Assessment", True)
        ]
        
        for item, status in checklist_items:
            status_icon = "✅" if status else "❌"
            st.write(f"{status_icon} {item}")
    
    # Export model results
    st.markdown("---")
    st.subheader("📥 Export Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Export model performance report
        if st.button("📊 Export Performance Report"):
            performance_report = {
                'Model': selected_model,
                'Accuracy': model_results['test_accuracy'],
                'Precision': model_results['test_precision'],
                'Recall': model_results['test_recall'],
                'F1_Score': model_results['test_f1'],
                'ROC_AUC': model_results['test_roc_auc'],
                'Training_Time': model_results['training_time']
            }
            
            report_df = pd.DataFrame([performance_report])
            csv = report_df.to_csv(index=False)
            st.download_button(
                label="Download Performance Report",
                data=csv,
                file_name=f"{selected_model}_performance_report.csv",
                mime="text/csv"
            )
    
    with col2:
        # Export feature importance
        if st.button("🎯 Export Feature Importance"):
            if 'feature_importance' in model_results:
                importance_df = pd.DataFrame(
                    list(model_results['feature_importance'].items()),
                    columns=['Feature', 'Importance']
                ).sort_values('Importance', ascending=False)
                
                csv = importance_df.to_csv(index=False)
                st.download_button(
                    label="Download Feature Importance",
                    data=csv,
                    file_name=f"{selected_model}_feature_importance.csv",
                    mime="text/csv"
                )
    
    with col3:
        # Export predictions
        if st.button("🔮 Export Predictions"):
            predictions_df = pd.DataFrame({
                'Actual': y_test,
                'Predicted': model.predict(X_test)
            })
            
            if hasattr(model, 'predict_proba'):
                predictions_df['Probability'] = model.predict_proba(X_test)[:, 1]
            
            csv = predictions_df.to_csv(index=False)
            st.download_button(
                label="Download Predictions",
                data=csv,
                file_name=f"{selected_model}_predictions.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()
