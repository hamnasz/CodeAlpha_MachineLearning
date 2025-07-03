import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                           roc_auc_score, classification_report, confusion_matrix)
from sklearn.feature_selection import RFE
import time
import warnings
warnings.filterwarnings('ignore')

class ModelTrainer:
    """
    Comprehensive model training utility for credit scoring.
    Handles multiple algorithms, hyperparameter tuning, and evaluation.
    """
    
    def __init__(self):
        self.models = {}
        self.best_params = {}
        self.training_results = {}
    
    def train_models(self, X, y, config):
        """
        Train multiple models with optional hyperparameter tuning
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
            config (dict): Training configuration
            
        Returns:
            dict: Training results including models, metrics, and data splits
        """
        # Split data
        test_size = config.get('test_size', 0.2)
        random_state = config.get('random_state', 42)
        stratify = y if config.get('stratify_split', True) else None
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify
        )
        
        # Initialize models
        models_to_train = config.get('models_to_train', ['Logistic Regression'])
        class_weight = config.get('class_weight', None)
        
        models = self._initialize_models(models_to_train, class_weight, random_state)
        
        # Training results storage
        trained_models = {}
        results = {}
        
        # Cross-validation settings
        cv_folds = config.get('cv_folds', 5)
        cv_scoring = config.get('cv_scoring', 'roc_auc')
        
        # Train each model
        for model_name, model in models.items():
            print(f"Training {model_name}...")
            start_time = time.time()
            
            try:
                # Hyperparameter tuning
                if config.get('perform_tuning', False) and model_name in config.get('param_grids', {}):
                    model = self._tune_hyperparameters(
                        model, X_train, y_train, 
                        config['param_grids'][model_name],
                        cv_folds, cv_scoring
                    )
                
                # Train model
                model.fit(X_train, y_train)
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_train, y_train, 
                                          cv=cv_folds, scoring=cv_scoring)
                
                # Predictions
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)
                
                # Calculate metrics
                metrics = self._calculate_metrics(
                    y_train, y_test, y_train_pred, y_test_pred, model, X_test
                )
                
                # Add timing and CV results
                metrics['training_time'] = time.time() - start_time
                metrics['cv_score_mean'] = cv_scores.mean()
                metrics['cv_score_std'] = cv_scores.std()
                
                # Feature importance (if available)
                if hasattr(model, 'feature_importances_'):
                    feature_importance = dict(zip(X.columns, model.feature_importances_))
                    metrics['feature_importance'] = feature_importance
                elif hasattr(model, 'coef_'):
                    feature_importance = dict(zip(X.columns, abs(model.coef_[0])))
                    metrics['feature_importance'] = feature_importance
                
                # Best parameters (if tuning was performed)
                if hasattr(model, 'best_params_'):
                    metrics['best_params'] = model.best_params_
                
                # Classification report
                metrics['classification_report'] = classification_report(y_test, y_test_pred)
                metrics['confusion_matrix'] = confusion_matrix(y_test, y_test_pred)
                
                # Store results
                trained_models[model_name] = model
                results[model_name] = metrics
                
                print(f"✓ {model_name} completed (ROC-AUC: {metrics['test_roc_auc']:.3f})")
                
            except Exception as e:
                print(f"✗ {model_name} failed: {str(e)}")
                continue
        
        return {
            'models': trained_models,
            'results': results,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
        }
    
    def _initialize_models(self, model_names, class_weight, random_state):
        """Initialize model instances"""
        models = {}
        
        for model_name in model_names:
            if model_name == 'Logistic Regression':
                models[model_name] = LogisticRegression(
                    random_state=random_state,
                    class_weight=class_weight,
                    max_iter=1000
                )
            
            elif model_name == 'Decision Tree':
                models[model_name] = DecisionTreeClassifier(
                    random_state=random_state,
                    class_weight=class_weight
                )
            
            elif model_name == 'Random Forest':
                models[model_name] = RandomForestClassifier(
                    random_state=random_state,
                    class_weight=class_weight,
                    n_estimators=100
                )
            
            elif model_name == 'Gradient Boosting':
                models[model_name] = GradientBoostingClassifier(
                    random_state=random_state,
                    n_estimators=100
                )
            
            elif model_name == 'SVM':
                models[model_name] = SVC(
                    random_state=random_state,
                    class_weight=class_weight,
                    probability=True
                )
        
        return models
    
    def _tune_hyperparameters(self, model, X_train, y_train, param_grid, cv_folds, scoring):
        """Perform hyperparameter tuning using GridSearchCV"""
        grid_search = GridSearchCV(
            model, param_grid, cv=cv_folds, scoring=scoring, n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        return grid_search
    
    def _calculate_metrics(self, y_train, y_test, y_train_pred, y_test_pred, model, X_test):
        """Calculate comprehensive evaluation metrics"""
        metrics = {}
        
        # Basic accuracy metrics
        metrics['train_accuracy'] = accuracy_score(y_train, y_train_pred)
        metrics['test_accuracy'] = accuracy_score(y_test, y_test_pred)
        
        # Precision, Recall, F1
        metrics['test_precision'] = precision_score(y_test, y_test_pred, average='binary')
        metrics['test_recall'] = recall_score(y_test, y_test_pred, average='binary')
        metrics['test_f1'] = f1_score(y_test, y_test_pred, average='binary')
        
        # ROC-AUC (if model supports probability prediction)
        if hasattr(model, 'predict_proba'):
            y_test_proba = model.predict_proba(X_test)[:, 1]
            metrics['test_roc_auc'] = roc_auc_score(y_test, y_test_proba)
        else:
            metrics['test_roc_auc'] = roc_auc_score(y_test, y_test_pred)
        
        return metrics
    
    def perform_feature_selection(self, X, y, model, n_features=10):
        """Perform recursive feature elimination"""
        rfe = RFE(model, n_features_to_select=n_features)
        X_selected = rfe.fit_transform(X, y)
        
        selected_features = X.columns[rfe.get_support()].tolist()
        feature_ranking = dict(zip(X.columns, rfe.ranking_))
        
        return X_selected, selected_features, feature_ranking
    
    def get_model_comparison(self, results):
        """Generate model comparison summary"""
        if not results:
            return pd.DataFrame()
        
        comparison_data = []
        for model_name, metrics in results.items():
            comparison_data.append({
                'Model': model_name,
                'CV_Score': f"{metrics['cv_score_mean']:.3f} (±{metrics['cv_score_std']:.3f})",
                'Test_Accuracy': f"{metrics['test_accuracy']:.3f}",
                'Test_Precision': f"{metrics['test_precision']:.3f}",
                'Test_Recall': f"{metrics['test_recall']:.3f}",
                'Test_F1': f"{metrics['test_f1']:.3f}",
                'Test_ROC_AUC': f"{metrics['test_roc_auc']:.3f}",
                'Training_Time': f"{metrics['training_time']:.2f}s"
            })
        
        return pd.DataFrame(comparison_data)
    
    def get_best_model(self, results, metric='test_roc_auc'):
        """Identify the best performing model"""
        if not results:
            return None, None
        
        best_model_name = max(results.keys(), key=lambda x: results[x][metric])
        best_score = results[best_model_name][metric]
        
        return best_model_name, best_score
    
    def save_models(self, models, filepath_prefix):
        """Save trained models to disk"""
        import pickle
        
        for model_name, model in models.items():
            # Clean model name for filename
            clean_name = model_name.lower().replace(' ', '_')
            filepath = f"{filepath_prefix}_{clean_name}.pkl"
            
            with open(filepath, 'wb') as f:
                pickle.dump(model, f)
    
    def load_model(self, filepath):
        """Load a trained model from disk"""
        import pickle
        
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        
        return model
    
    def get_training_recommendations(self, results, X, y):
        """Generate training recommendations based on results"""
        recommendations = []
        
        if not results:
            recommendations.append("No models were successfully trained. Check data quality and configuration.")
            return recommendations
        
        # Check for overfitting
        for model_name, metrics in results.items():
            train_acc = metrics.get('train_accuracy', 0)
            test_acc = metrics.get('test_accuracy', 0)
            
            if train_acc - test_acc > 0.1:
                recommendations.append(f"{model_name}: Potential overfitting detected. Consider regularization or more data.")
        
        # Check for low performance
        best_auc = max(metrics['test_roc_auc'] for metrics in results.values())
        if best_auc < 0.7:
            recommendations.append("Overall model performance is low. Consider feature engineering or different algorithms.")
        
        # Check class imbalance
        class_balance = y.value_counts().min() / y.value_counts().max()
        if class_balance < 0.3:
            recommendations.append("Class imbalance detected. Consider using balanced class weights or resampling techniques.")
        
        # Check feature count
        if len(X.columns) > 50:
            recommendations.append("High number of features. Consider feature selection to reduce complexity.")
        
        # Performance recommendations
        if best_auc >= 0.8:
            recommendations.append("Good model performance achieved. Consider ensemble methods for further improvement.")
        
        return recommendations
    
    def create_ensemble_model(self, models, X_test, y_test, method='voting'):
        """Create an ensemble of trained models"""
        from sklearn.ensemble import VotingClassifier
        
        if len(models) < 2:
            return None
        
        # Prepare estimators for ensemble
        estimators = [(name, model) for name, model in models.items()]
        
        if method == 'voting':
            ensemble = VotingClassifier(estimators=estimators, voting='soft')
        else:
            # Could implement other ensemble methods here
            ensemble = VotingClassifier(estimators=estimators, voting='soft')
        
        # Note: In practice, you'd need to refit the ensemble on training data
        # This is a simplified implementation
        
        return ensemble
