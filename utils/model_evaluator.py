import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, precision_recall_curve, confusion_matrix, classification_report,
    brier_score_loss
)
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

class ModelEvaluator:
    """
    Comprehensive model evaluation utility for credit scoring models.
    Provides detailed performance analysis, interpretability, and business metrics.
    """
    
    def __init__(self):
        self.evaluation_results = {}
        self.business_metrics = {}
    
    def evaluate_model(self, model, X_test, y_test, model_name="Model"):
        """
        Comprehensive model evaluation
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test targets
            model_name: Name for the model
            
        Returns:
            dict: Comprehensive evaluation results
        """
        results = {}
        
        # Get predictions
        y_pred = model.predict(X_test)
        y_pred_proba = None
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Basic metrics
        results['basic_metrics'] = self._calculate_basic_metrics(y_test, y_pred, y_pred_proba)
        
        # Confusion matrix analysis
        results['confusion_matrix'] = self._analyze_confusion_matrix(y_test, y_pred)
        
        # ROC and PR curves
        if y_pred_proba is not None:
            results['roc_analysis'] = self._analyze_roc_curve(y_test, y_pred_proba)
            results['pr_analysis'] = self._analyze_pr_curve(y_test, y_pred_proba)
            results['calibration_analysis'] = self._analyze_calibration(y_test, y_pred_proba)
        
        # Feature importance analysis
        results['feature_importance'] = self._analyze_feature_importance(model, X_test, y_test)
        
        # Threshold analysis
        if y_pred_proba is not None:
            results['threshold_analysis'] = self._analyze_thresholds(y_test, y_pred_proba)
        
        # Model interpretation
        results['interpretation'] = self._generate_interpretation(results, model_name)
        
        self.evaluation_results[model_name] = results
        return results
    
    def _calculate_basic_metrics(self, y_true, y_pred, y_pred_proba=None):
        """Calculate basic classification metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='binary'),
            'recall': recall_score(y_true, y_pred, average='binary'),
            'f1_score': f1_score(y_true, y_pred, average='binary'),
            'specificity': self._calculate_specificity(y_true, y_pred)
        }
        
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            metrics['brier_score'] = brier_score_loss(y_true, y_pred_proba)
        else:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred)
        
        return metrics
    
    def _calculate_specificity(self, y_true, y_pred):
        """Calculate specificity (True Negative Rate)"""
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        return tn / (tn + fp) if (tn + fp) > 0 else 0
    
    def _analyze_confusion_matrix(self, y_true, y_pred):
        """Analyze confusion matrix in detail"""
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        total = len(y_true)
        
        analysis = {
            'matrix': cm,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn,
            'true_positives': tp,
            'error_rate': (fp + fn) / total,
            'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
            'false_negative_rate': fn / (fn + tp) if (fn + tp) > 0 else 0,
            'positive_predictive_value': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'negative_predictive_value': tn / (tn + fn) if (tn + fn) > 0 else 0
        }
        
        return analysis
    
    def _analyze_roc_curve(self, y_true, y_pred_proba):
        """Analyze ROC curve characteristics"""
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        auc_score = roc_auc_score(y_true, y_pred_proba)
        
        # Find optimal threshold (closest to top-left corner)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        
        analysis = {
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds,
            'auc': auc_score,
            'optimal_threshold': optimal_threshold,
            'optimal_tpr': tpr[optimal_idx],
            'optimal_fpr': fpr[optimal_idx]
        }
        
        return analysis
    
    def _analyze_pr_curve(self, y_true, y_pred_proba):
        """Analyze Precision-Recall curve"""
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        
        # Calculate AUC for PR curve
        pr_auc = np.trapz(precision, recall)
        
        # Find threshold that maximizes F1 score
        f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1])
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx]
        
        analysis = {
            'precision': precision,
            'recall': recall,
            'thresholds': thresholds,
            'pr_auc': pr_auc,
            'optimal_threshold_f1': optimal_threshold,
            'optimal_precision': precision[optimal_idx],
            'optimal_recall': recall[optimal_idx],
            'max_f1_score': f1_scores[optimal_idx]
        }
        
        return analysis
    
    def _analyze_calibration(self, y_true, y_pred_proba):
        """Analyze model calibration"""
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_pred_proba, n_bins=10
        )
        
        # Calculate calibration metrics
        calibration_error = np.mean(np.abs(fraction_of_positives - mean_predicted_value))
        
        analysis = {
            'fraction_of_positives': fraction_of_positives,
            'mean_predicted_value': mean_predicted_value,
            'calibration_error': calibration_error,
            'brier_score': brier_score_loss(y_true, y_pred_proba)
        }
        
        return analysis
    
    def _analyze_feature_importance(self, model, X_test, y_test):
        """Analyze feature importance using various methods"""
        analysis = {}
        
        # Built-in feature importance (if available)
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            analysis['built_in'] = dict(zip(X_test.columns, importance))
        
        # Coefficient-based importance (for linear models)
        if hasattr(model, 'coef_'):
            importance = np.abs(model.coef_[0])
            analysis['coefficients'] = dict(zip(X_test.columns, importance))
        
        # Permutation importance
        try:
            perm_importance = permutation_importance(
                model, X_test, y_test, n_repeats=5, random_state=42
            )
            analysis['permutation'] = dict(zip(X_test.columns, perm_importance.importances_mean))
        except Exception as e:
            print(f"Permutation importance calculation failed: {str(e)}")
        
        return analysis
    
    def _analyze_thresholds(self, y_true, y_pred_proba):
        """Analyze performance across different thresholds"""
        thresholds = np.arange(0.1, 1.0, 0.05)
        threshold_analysis = []
        
        for threshold in thresholds:
            y_pred_thresh = (y_pred_proba >= threshold).astype(int)
            
            try:
                precision = precision_score(y_true, y_pred_thresh)
                recall = recall_score(y_true, y_pred_thresh)
                f1 = f1_score(y_true, y_pred_thresh)
                accuracy = accuracy_score(y_true, y_pred_thresh)
                
                # Business metrics
                cm = confusion_matrix(y_true, y_pred_thresh)
                tn, fp, fn, tp = cm.ravel()
                
                approval_rate = (tp + fp) / len(y_true)
                default_rate = fp / (tp + fp) if (tp + fp) > 0 else 0
                
                threshold_analysis.append({
                    'threshold': threshold,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'accuracy': accuracy,
                    'approval_rate': approval_rate,
                    'default_rate': default_rate,
                    'true_positives': tp,
                    'false_positives': fp,
                    'true_negatives': tn,
                    'false_negatives': fn
                })
            
            except Exception:
                continue
        
        return pd.DataFrame(threshold_analysis)
    
    def _generate_interpretation(self, results, model_name):
        """Generate human-readable interpretation of results"""
        interpretation = {
            'summary': '',
            'strengths': [],
            'weaknesses': [],
            'recommendations': []
        }
        
        metrics = results['basic_metrics']
        
        # Overall performance assessment
        auc = metrics['roc_auc']
        if auc >= 0.9:
            performance_level = "Excellent"
        elif auc >= 0.8:
            performance_level = "Good"
        elif auc >= 0.7:
            performance_level = "Fair"
        else:
            performance_level = "Poor"
        
        interpretation['summary'] = f"{model_name} shows {performance_level.lower()} performance with an AUC of {auc:.3f}."
        
        # Identify strengths
        if metrics['precision'] >= 0.8:
            interpretation['strengths'].append("High precision - model is good at avoiding false positives")
        
        if metrics['recall'] >= 0.8:
            interpretation['strengths'].append("High recall - model is good at identifying positive cases")
        
        if metrics['specificity'] >= 0.8:
            interpretation['strengths'].append("High specificity - model is good at identifying negative cases")
        
        # Identify weaknesses
        if metrics['precision'] < 0.7:
            interpretation['weaknesses'].append("Low precision - model may approve too many risky applicants")
        
        if metrics['recall'] < 0.7:
            interpretation['weaknesses'].append("Low recall - model may reject too many good applicants")
        
        if metrics['specificity'] < 0.7:
            interpretation['weaknesses'].append("Low specificity - model may not identify good applicants well")
        
        # Generate recommendations
        if auc < 0.7:
            interpretation['recommendations'].append("Consider feature engineering or different algorithms")
        
        if 'calibration_analysis' in results:
            cal_error = results['calibration_analysis']['calibration_error']
            if cal_error > 0.1:
                interpretation['recommendations'].append("Model calibration could be improved")
        
        # Business-specific recommendations
        if metrics['precision'] < 0.7:
            interpretation['recommendations'].append("Consider stricter approval criteria to reduce default risk")
        
        if metrics['recall'] < 0.7:
            interpretation['recommendations'].append("Consider looser criteria to capture more good applicants")
        
        return interpretation
    
    def calculate_business_impact(self, y_true, y_pred_proba, threshold=0.5, 
                                default_cost=10000, loan_profit=2000, 
                                processing_cost=100, opportunity_cost=500):
        """Calculate business impact metrics"""
        y_pred = (y_pred_proba >= threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Calculate costs and profits
        total_default_cost = fp * default_cost  # False positives result in defaults
        total_processing_cost = (tp + fp) * processing_cost  # All approvals incur processing cost
        total_opportunity_cost = fn * opportunity_cost  # False negatives lose opportunities
        total_profit = tp * loan_profit  # True positives generate profit
        
        net_profit = total_profit - total_default_cost - total_processing_cost - total_opportunity_cost
        
        business_metrics = {
            'total_profit': total_profit,
            'total_default_cost': total_default_cost,
            'total_processing_cost': total_processing_cost,
            'total_opportunity_cost': total_opportunity_cost,
            'net_profit': net_profit,
            'profit_per_customer': net_profit / len(y_true),
            'approval_rate': (tp + fp) / len(y_true),
            'default_rate': fp / (tp + fp) if (tp + fp) > 0 else 0,
            'capture_rate': tp / (tp + fn) if (tp + fn) > 0 else 0
        }
        
        return business_metrics
    
    def compare_models(self, model_results):
        """Compare multiple model evaluation results"""
        if len(model_results) < 2:
            return "Need at least 2 models for comparison"
        
        comparison = pd.DataFrame()
        
        for model_name, results in model_results.items():
            metrics = results['basic_metrics']
            comparison[model_name] = [
                metrics['accuracy'],
                metrics['precision'],
                metrics['recall'],
                metrics['f1_score'],
                metrics['roc_auc'],
                metrics['specificity']
            ]
        
        comparison.index = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'Specificity']
        
        # Add ranking
        rankings = {}
        for metric in comparison.index:
            rankings[metric] = comparison.loc[metric].rank(ascending=False).astype(int)
        
        ranking_df = pd.DataFrame(rankings).T
        
        return {
            'metrics_comparison': comparison,
            'rankings': ranking_df,
            'best_overall': comparison.mean().idxmax()
        }
    
    def generate_model_report(self, model_name):
        """Generate a comprehensive model report"""
        if model_name not in self.evaluation_results:
            return "Model not found in evaluation results"
        
        results = self.evaluation_results[model_name]
        
        report = f"""
        MODEL EVALUATION REPORT: {model_name}
        =====================================
        
        PERFORMANCE SUMMARY:
        - Accuracy: {results['basic_metrics']['accuracy']:.3f}
        - Precision: {results['basic_metrics']['precision']:.3f}
        - Recall: {results['basic_metrics']['recall']:.3f}
        - F1-Score: {results['basic_metrics']['f1_score']:.3f}
        - ROC-AUC: {results['basic_metrics']['roc_auc']:.3f}
        - Specificity: {results['basic_metrics']['specificity']:.3f}
        
        INTERPRETATION:
        {results['interpretation']['summary']}
        
        STRENGTHS:
        """
        
        for strength in results['interpretation']['strengths']:
            report += f"- {strength}\n"
        
        report += "\nWEAKNESSES:\n"
        for weakness in results['interpretation']['weaknesses']:
            report += f"- {weakness}\n"
        
        report += "\nRECOMMENDATIONS:\n"
        for recommendation in results['interpretation']['recommendations']:
            report += f"- {recommendation}\n"
        
        return report
    
    def save_evaluation_results(self, filepath):
        """Save evaluation results to file"""
        import pickle
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.evaluation_results, f)
    
    def load_evaluation_results(self, filepath):
        """Load evaluation results from file"""
        import pickle
        
        with open(filepath, 'rb') as f:
            self.evaluation_results = pickle.load(f)
