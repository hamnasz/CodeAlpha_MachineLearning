import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

class BiasDetector:
    """
    Comprehensive bias detection utility for credit scoring models.
    Implements various fairness metrics to detect discriminatory bias.
    """
    
    def __init__(self):
        self.bias_results = {}
        self.fairness_thresholds = {
            'demographic_parity': 0.1,  # Difference in approval rates
            'equalized_odds': 0.1,      # Difference in TPR/FPR
            'calibration': 0.1,         # Difference in calibration
            'treatment_equality': 0.1    # Difference in error rates
        }
    
    def detect_bias(self, X_test, y_true, y_pred, protected_attributes, y_pred_proba=None):
        """
        Detect bias across protected attributes
        
        Args:
            X_test: Test features
            y_true: True labels
            y_pred: Predicted labels
            protected_attributes: List of protected attribute column names
            y_pred_proba: Predicted probabilities (optional)
            
        Returns:
            dict: Bias analysis results for each protected attribute
        """
        bias_results = {}
        
        for attr in protected_attributes:
            if attr not in X_test.columns:
                continue
            
            attr_bias = self._analyze_attribute_bias(
                X_test[attr], y_true, y_pred, y_pred_proba, attr
            )
            bias_results[attr] = attr_bias
        
        self.bias_results = bias_results
        return bias_results
    
    def _analyze_attribute_bias(self, protected_attr, y_true, y_pred, y_pred_proba, attr_name):
        """Analyze bias for a single protected attribute"""
        results = {}
        
        # Get unique groups in the protected attribute
        groups = protected_attr.unique()
        
        if len(groups) < 2:
            return {'error': f'Insufficient groups in {attr_name} for bias analysis'}
        
        # Demographic Parity (Statistical Parity)
        results['demographic_parity'] = self._calculate_demographic_parity(
            protected_attr, y_pred, groups
        )
        
        # Equalized Odds
        results['equalized_odds'] = self._calculate_equalized_odds(
            protected_attr, y_true, y_pred, groups
        )
        
        # Equal Opportunity (True Positive Rate equality)
        results['equal_opportunity'] = self._calculate_equal_opportunity(
            protected_attr, y_true, y_pred, groups
        )
        
        # Calibration (if probabilities available)
        if y_pred_proba is not None:
            results['calibration'] = self._calculate_calibration_bias(
                protected_attr, y_true, y_pred_proba, groups
            )
        
        # Treatment Equality
        results['treatment_equality'] = self._calculate_treatment_equality(
            protected_attr, y_true, y_pred, groups
        )
        
        # Overall bias assessment
        results['bias_assessment'] = self._assess_overall_bias(results, attr_name)
        
        return results
    
    def _calculate_demographic_parity(self, protected_attr, y_pred, groups):
        """
        Calculate demographic parity (statistical parity)
        Measures difference in approval rates between groups
        """
        approval_rates = {}
        
        for group in groups:
            group_mask = protected_attr == group
            group_predictions = y_pred[group_mask]
            approval_rate = np.mean(group_predictions)
            approval_rates[group] = approval_rate
        
        # Calculate maximum difference between groups
        rates = list(approval_rates.values())
        max_diff = max(rates) - min(rates)
        
        # Calculate ratio (for multiplicative fairness)
        min_rate = min(rates)
        max_rate = max(rates)
        ratio = min_rate / max_rate if max_rate > 0 else 0
        
        return {
            'approval_rates': approval_rates,
            'max_difference': max_diff,
            'ratio': ratio,
            'is_fair': max_diff <= self.fairness_thresholds['demographic_parity']
        }
    
    def _calculate_equalized_odds(self, protected_attr, y_true, y_pred, groups):
        """
        Calculate equalized odds
        Measures difference in TPR and FPR between groups
        """
        group_metrics = {}
        
        for group in groups:
            group_mask = protected_attr == group
            group_true = y_true[group_mask]
            group_pred = y_pred[group_mask]
            
            if len(group_true) == 0:
                continue
            
            # Calculate TPR and FPR
            cm = confusion_matrix(group_true, group_pred)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                
                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                
                group_metrics[group] = {'tpr': tpr, 'fpr': fpr}
        
        # Calculate differences
        if len(group_metrics) >= 2:
            tpr_values = [metrics['tpr'] for metrics in group_metrics.values()]
            fpr_values = [metrics['fpr'] for metrics in group_metrics.values()]
            
            tpr_diff = max(tpr_values) - min(tpr_values)
            fpr_diff = max(fpr_values) - min(fpr_values)
            
            max_diff = max(tpr_diff, fpr_diff)
            
            return {
                'group_metrics': group_metrics,
                'tpr_difference': tpr_diff,
                'fpr_difference': fpr_diff,
                'max_difference': max_diff,
                'is_fair': max_diff <= self.fairness_thresholds['equalized_odds']
            }
        
        return {'error': 'Insufficient data for equalized odds calculation'}
    
    def _calculate_equal_opportunity(self, protected_attr, y_true, y_pred, groups):
        """
        Calculate equal opportunity
        Measures difference in TPR (recall) between groups
        """
        tpr_by_group = {}
        
        for group in groups:
            group_mask = protected_attr == group
            group_true = y_true[group_mask]
            group_pred = y_pred[group_mask]
            
            if len(group_true) == 0:
                continue
            
            # Calculate TPR for positive class
            positive_mask = group_true == 1
            if positive_mask.sum() > 0:
                tpr = np.mean(group_pred[positive_mask])
                tpr_by_group[group] = tpr
        
        if len(tpr_by_group) >= 2:
            tpr_values = list(tpr_by_group.values())
            max_diff = max(tpr_values) - min(tpr_values)
            
            return {
                'tpr_by_group': tpr_by_group,
                'difference': max_diff,
                'is_fair': max_diff <= self.fairness_thresholds['equalized_odds']
            }
        
        return {'error': 'Insufficient data for equal opportunity calculation'}
    
    def _calculate_calibration_bias(self, protected_attr, y_true, y_pred_proba, groups):
        """
        Calculate calibration bias
        Measures difference in calibration between groups
        """
        calibration_by_group = {}
        
        for group in groups:
            group_mask = protected_attr == group
            group_true = y_true[group_mask]
            group_proba = y_pred_proba[group_mask]
            
            if len(group_true) == 0:
                continue
            
            # Calculate calibration error
            # Bin predictions and calculate mean predicted vs actual
            try:
                bins = np.linspace(0, 1, 11)
                bin_indices = np.digitize(group_proba, bins) - 1
                
                calibration_error = 0
                for bin_idx in range(len(bins) - 1):
                    bin_mask = bin_indices == bin_idx
                    if bin_mask.sum() > 0:
                        mean_predicted = np.mean(group_proba[bin_mask])
                        mean_actual = np.mean(group_true[bin_mask])
                        calibration_error += abs(mean_predicted - mean_actual) * bin_mask.sum()
                
                calibration_error /= len(group_true)
                calibration_by_group[group] = calibration_error
                
            except Exception:
                # Simple calibration: mean predicted vs mean actual
                calibration_error = abs(np.mean(group_proba) - np.mean(group_true))
                calibration_by_group[group] = calibration_error
        
        if len(calibration_by_group) >= 2:
            cal_values = list(calibration_by_group.values())
            max_diff = max(cal_values) - min(cal_values)
            
            return {
                'calibration_by_group': calibration_by_group,
                'difference': max_diff,
                'is_fair': max_diff <= self.fairness_thresholds['calibration']
            }
        
        return {'error': 'Insufficient data for calibration analysis'}
    
    def _calculate_treatment_equality(self, protected_attr, y_true, y_pred, groups):
        """
        Calculate treatment equality
        Measures difference in error rates between groups
        """
        error_rates = {}
        
        for group in groups:
            group_mask = protected_attr == group
            group_true = y_true[group_mask]
            group_pred = y_pred[group_mask]
            
            if len(group_true) == 0:
                continue
            
            # Calculate error rate
            errors = group_true != group_pred
            error_rate = np.mean(errors)
            error_rates[group] = error_rate
        
        if len(error_rates) >= 2:
            error_values = list(error_rates.values())
            max_diff = max(error_values) - min(error_values)
            
            return {
                'error_rates': error_rates,
                'difference': max_diff,
                'is_fair': max_diff <= self.fairness_thresholds['treatment_equality']
            }
        
        return {'error': 'Insufficient data for treatment equality calculation'}
    
    def _assess_overall_bias(self, results, attr_name):
        """Assess overall bias for an attribute"""
        bias_indicators = []
        
        # Check each fairness metric
        for metric_name, metric_results in results.items():
            if isinstance(metric_results, dict) and 'is_fair' in metric_results:
                if not metric_results['is_fair']:
                    bias_indicators.append(metric_name)
        
        if not bias_indicators:
            assessment = "No significant bias detected"
            risk_level = "Low"
        elif len(bias_indicators) == 1:
            assessment = f"Potential bias detected in {bias_indicators[0]}"
            risk_level = "Medium"
        else:
            assessment = f"Multiple bias indicators: {', '.join(bias_indicators)}"
            risk_level = "High"
        
        return {
            'assessment': assessment,
            'risk_level': risk_level,
            'bias_indicators': bias_indicators,
            'total_metrics_checked': len([k for k in results.keys() if k != 'bias_assessment'])
        }
    
    def generate_bias_report(self, attribute_name=None):
        """Generate a comprehensive bias report"""
        if not self.bias_results:
            return "No bias analysis results available. Run detect_bias() first."
        
        if attribute_name and attribute_name not in self.bias_results:
            return f"No bias analysis results for attribute '{attribute_name}'"
        
        # Generate report for specific attribute or all attributes
        attributes_to_report = [attribute_name] if attribute_name else self.bias_results.keys()
        
        report = "BIAS DETECTION REPORT\n"
        report += "=" * 50 + "\n\n"
        
        for attr in attributes_to_report:
            results = self.bias_results[attr]
            
            if 'error' in results:
                report += f"ATTRIBUTE: {attr}\n"
                report += f"ERROR: {results['error']}\n\n"
                continue
            
            assessment = results.get('bias_assessment', {})
            
            report += f"ATTRIBUTE: {attr}\n"
            report += f"Overall Assessment: {assessment.get('assessment', 'Unknown')}\n"
            report += f"Risk Level: {assessment.get('risk_level', 'Unknown')}\n\n"
            
            # Demographic Parity
            if 'demographic_parity' in results:
                dp = results['demographic_parity']
                report += f"Demographic Parity:\n"
                report += f"  - Max difference in approval rates: {dp.get('max_difference', 0):.3f}\n"
                report += f"  - Fair: {dp.get('is_fair', False)}\n\n"
            
            # Equalized Odds
            if 'equalized_odds' in results:
                eo = results['equalized_odds']
                if 'error' not in eo:
                    report += f"Equalized Odds:\n"
                    report += f"  - Max difference: {eo.get('max_difference', 0):.3f}\n"
                    report += f"  - Fair: {eo.get('is_fair', False)}\n\n"
            
            # Calibration
            if 'calibration' in results:
                cal = results['calibration']
                if 'error' not in cal:
                    report += f"Calibration:\n"
                    report += f"  - Max difference: {cal.get('difference', 0):.3f}\n"
                    report += f"  - Fair: {cal.get('is_fair', False)}\n\n"
            
            report += "-" * 30 + "\n"
        
        return report
    
    def get_bias_summary(self):
        """Get summary of bias detection across all attributes"""
        if not self.bias_results:
            return {}
        
        summary = {
            'total_attributes_analyzed': len(self.bias_results),
            'high_risk_attributes': [],
            'medium_risk_attributes': [],
            'low_risk_attributes': [],
            'overall_risk_level': 'Low'
        }
        
        for attr, results in self.bias_results.items():
            if 'bias_assessment' in results:
                risk_level = results['bias_assessment'].get('risk_level', 'Unknown')
                
                if risk_level == 'High':
                    summary['high_risk_attributes'].append(attr)
                elif risk_level == 'Medium':
                    summary['medium_risk_attributes'].append(attr)
                else:
                    summary['low_risk_attributes'].append(attr)
        
        # Determine overall risk level
        if summary['high_risk_attributes']:
            summary['overall_risk_level'] = 'High'
        elif summary['medium_risk_attributes']:
            summary['overall_risk_level'] = 'Medium'
        
        return summary
    
    def suggest_bias_mitigation(self, attribute_name):
        """Suggest bias mitigation strategies for a specific attribute"""
        if attribute_name not in self.bias_results:
            return f"No bias analysis available for {attribute_name}"
        
        results = self.bias_results[attribute_name]
        suggestions = []
        
        # Check which metrics show bias
        if 'demographic_parity' in results and not results['demographic_parity'].get('is_fair', True):
            suggestions.append(
                "Demographic Parity violation detected. Consider:\n"
                "- Adjusting approval thresholds by group\n"
                "- Using fairness-aware algorithms\n"
                "- Post-processing to equalize approval rates"
            )
        
        if 'equalized_odds' in results and not results['equalized_odds'].get('is_fair', True):
            suggestions.append(
                "Equalized Odds violation detected. Consider:\n"
                "- Using adversarial debiasing techniques\n"
                "- Implementing equalized odds post-processing\n"
                "- Reweighting training data"
            )
        
        if 'calibration' in results and not results['calibration'].get('is_fair', True):
            suggestions.append(
                "Calibration bias detected. Consider:\n"
                "- Separate calibration by group\n"
                "- Using calibration-aware training\n"
                "- Post-hoc calibration methods"
            )
        
        if not suggestions:
            suggestions.append("No significant bias detected. Continue monitoring.")
        
        return "\n\n".join(suggestions)
    
    def save_bias_results(self, filepath):
        """Save bias detection results"""
        import pickle
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.bias_results, f)
    
    def load_bias_results(self, filepath):
        """Load bias detection results"""
        import pickle
        
        with open(filepath, 'rb') as f:
            self.bias_results = pickle.load(f)
