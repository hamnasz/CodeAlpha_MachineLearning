import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
from sklearn.linear_model import LogisticRegression
import re
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    """
    Comprehensive feature engineering utility for credit scoring models.
    Creates derived features, handles encoding, scaling, and feature selection.
    """
    
    def __init__(self):
        self.encoders = {}
        self.scalers = {}
        self.feature_selectors = {}
        self.feature_names = []
        self.engineering_report = {}
    
    def engineer_features(self, data, config):
        """
        Main feature engineering pipeline
        
        Args:
            data (pd.DataFrame): Preprocessed data
            config (dict): Configuration parameters for feature engineering
            
        Returns:
            tuple: (engineered_data, engineering_report)
        """
        engineered_data = data.copy()
        report = {'new_features': [], 'encoded_features': {}, 'scaled_features': [], 'selected_features': []}
        
        # 1. Create derived features
        engineered_data, new_features = self._create_derived_features(engineered_data, config)
        if new_features:
            report['new_features'] = new_features
        
        # 2. Handle custom mathematical features
        if config.get('custom_features'):
            engineered_data, custom_features = self._create_custom_features(
                engineered_data, config['custom_features']
            )
            if custom_features:
                report['new_features'].extend(custom_features)
        
        # 3. Encode categorical variables
        if config.get('categorical_to_encode'):
            engineered_data, encoding_report = self._encode_categorical_features(
                engineered_data, config
            )
            if encoding_report:
                report['encoded_features'] = encoding_report
        
        # 4. Scale numerical features
        if config.get('features_to_scale'):
            engineered_data, scaled_features = self._scale_features(
                engineered_data, config
            )
            if scaled_features:
                report['scaled_features'] = scaled_features
        
        # 5. Feature selection
        if config.get('perform_selection', False):
            engineered_data, selection_report = self._select_features(
                engineered_data, config
            )
            if selection_report:
                report.update(selection_report)
        
        return engineered_data, report
    
    def _create_derived_features(self, data, config):
        """Create derived financial and demographic features"""
        new_features = []
        
        # Financial ratio features
        if config.get('create_debt_income', False):
            if any('income' in col.lower() for col in data.columns) and any('debt' in col.lower() for col in data.columns):
                income_cols = [col for col in data.columns if 'income' in col.lower()]
                debt_cols = [col for col in data.columns if 'debt' in col.lower()]
                
                if income_cols and debt_cols:
                    income_col = income_cols[0]
                    debt_col = debt_cols[0]
                    
                    # Avoid division by zero
                    data['debt_to_income_ratio'] = np.where(
                        data[income_col] > 0,
                        data[debt_col] / data[income_col],
                        0
                    )
                    new_features.append('debt_to_income_ratio')
        
        if config.get('create_payment_ratio', False):
            payment_cols = [col for col in data.columns if 'payment' in col.lower()]
            income_cols = [col for col in data.columns if 'income' in col.lower()]
            
            if payment_cols and income_cols:
                payment_col = payment_cols[0]
                income_col = income_cols[0]
                
                data['payment_to_income_ratio'] = np.where(
                    data[income_col] > 0,
                    data[payment_col] / data[income_col],
                    0
                )
                new_features.append('payment_to_income_ratio')
        
        if config.get('create_utilization_squared', False):
            utilization_cols = [col for col in data.columns if 'utilization' in col.lower()]
            if utilization_cols:
                util_col = utilization_cols[0]
                data['credit_utilization_squared'] = data[util_col] ** 2
                new_features.append('credit_utilization_squared')
        
        # Age-based features
        if config.get('create_age_bins', False):
            age_cols = [col for col in data.columns if 'age' in col.lower()]
            if age_cols:
                age_col = age_cols[0]
                data['age_group'] = pd.cut(
                    data[age_col],
                    bins=[0, 25, 35, 50, 65, 100],
                    labels=['Young', 'Adult', 'Middle_Age', 'Senior', 'Elderly']
                )
                new_features.append('age_group')
        
        if config.get('create_age_income', False):
            age_cols = [col for col in data.columns if 'age' in col.lower()]
            income_cols = [col for col in data.columns if 'income' in col.lower()]
            
            if age_cols and income_cols:
                age_col = age_cols[0]
                income_col = income_cols[0]
                
                # Normalize age and income, then create interaction
                age_norm = (data[age_col] - data[age_col].mean()) / data[age_col].std()
                income_norm = (data[income_col] - data[income_col].mean()) / data[income_col].std()
                data['age_income_interaction'] = age_norm * income_norm
                new_features.append('age_income_interaction')
        
        # Payment history features
        if config.get('create_payment_severity', False):
            payment_cols = [col for col in data.columns if 'payment' in col.lower() and 'history' in col.lower()]
            if not payment_cols:
                payment_cols = [col for col in data.columns if 'payment' in col.lower()]
            
            if payment_cols:
                payment_col = payment_cols[0]
                
                # Create payment severity score
                data['payment_severity_score'] = np.where(
                    data[payment_col] == 0, 0,
                    np.where(data[payment_col] <= 2, 1,
                           np.where(data[payment_col] <= 5, 2, 3))
                )
                new_features.append('payment_severity_score')
        
        # Risk scoring
        if config.get('create_risk_score', False):
            # Create a composite risk score based on available features
            risk_components = []
            
            # Debt-to-income component
            if 'debt_to_income_ratio' in data.columns:
                risk_components.append(data['debt_to_income_ratio'])
            
            # Credit utilization component
            utilization_cols = [col for col in data.columns if 'utilization' in col.lower()]
            if utilization_cols:
                risk_components.append(data[utilization_cols[0]])
            
            # Payment history component
            payment_cols = [col for col in data.columns if 'payment' in col.lower()]
            if payment_cols:
                # Normalize payment history (higher payments = higher risk)
                payment_col = payment_cols[0]
                max_payment = data[payment_col].max()
                if max_payment > 0:
                    risk_components.append(data[payment_col] / max_payment)
            
            if risk_components:
                # Average the risk components
                risk_array = np.column_stack(risk_components)
                data['composite_risk_score'] = np.mean(risk_array, axis=1)
                new_features.append('composite_risk_score')
        
        # Income brackets
        if config.get('create_income_bins', False):
            income_cols = [col for col in data.columns if 'income' in col.lower()]
            if income_cols:
                income_col = income_cols[0]
                
                # Create income brackets based on quartiles
                data['income_bracket'] = pd.qcut(
                    data[income_col],
                    q=4,
                    labels=['Low_Income', 'Lower_Middle', 'Upper_Middle', 'High_Income']
                )
                new_features.append('income_bracket')
        
        return data, new_features
    
    def _create_custom_features(self, data, custom_features_text):
        """Create custom mathematical features from user input"""
        new_features = []
        
        if not custom_features_text.strip():
            return data, new_features
        
        lines = custom_features_text.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line or '=' not in line:
                continue
            
            try:
                feature_name, expression = line.split('=', 1)
                feature_name = feature_name.strip()
                expression = expression.strip()
                
                # Safe evaluation of mathematical expressions
                # Replace column names with data references
                safe_expression = expression
                for col in data.columns:
                    if col in expression:
                        safe_expression = safe_expression.replace(col, f"data['{col}']")
                
                # Replace mathematical functions
                safe_expression = safe_expression.replace('log(', 'np.log(')
                safe_expression = safe_expression.replace('sqrt(', 'np.sqrt(')
                safe_expression = safe_expression.replace('exp(', 'np.exp(')
                safe_expression = safe_expression.replace('abs(', 'np.abs(')
                
                # Evaluate the expression
                result = eval(safe_expression)
                
                # Handle infinite and NaN values
                if isinstance(result, pd.Series):
                    result = result.replace([np.inf, -np.inf], np.nan)
                    result = result.fillna(0)
                
                data[feature_name] = result
                new_features.append(feature_name)
                
            except Exception as e:
                # Skip invalid expressions
                print(f"Warning: Could not create feature '{feature_name}': {str(e)}")
                continue
        
        return data, new_features
    
    def _encode_categorical_features(self, data, config):
        """Encode categorical variables"""
        encoding_report = {}
        method = config.get('encoding_method', 'onehot')
        categorical_cols = config.get('categorical_to_encode', [])
        
        for col in categorical_cols:
            if col not in data.columns:
                continue
            
            original_col = col
            
            if method == 'onehot':
                # One-hot encoding
                max_categories = config.get('max_categories', 10)
                
                # Limit categories if too many
                value_counts = data[col].value_counts()
                if len(value_counts) > max_categories:
                    # Keep top categories, group others as "Other"
                    top_categories = value_counts.head(max_categories - 1).index
                    data[col] = data[col].apply(
                        lambda x: x if x in top_categories else 'Other'
                    )
                
                # Perform one-hot encoding
                dummies = pd.get_dummies(
                    data[col], 
                    prefix=col,
                    drop_first=config.get('drop_first', True)
                )
                
                # Add dummy columns to data
                data = pd.concat([data, dummies], axis=1)
                data = data.drop(columns=[col])
                
                encoding_report[original_col] = list(dummies.columns)
            
            elif method == 'label':
                # Label encoding
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col].astype(str))
                self.encoders[col] = le
                encoding_report[original_col] = f"{col}_encoded"
            
            elif method == 'target':
                # Target encoding (mean encoding)
                if 'target' in data.columns:
                    target_means = data.groupby(col)['target'].mean()
                    data[f'{col}_target_encoded'] = data[col].map(target_means)
                    data = data.drop(columns=[col])
                    encoding_report[original_col] = f"{col}_target_encoded"
            
            elif method == 'frequency':
                # Frequency encoding
                frequency_map = data[col].value_counts().to_dict()
                data[f'{col}_frequency'] = data[col].map(frequency_map)
                data = data.drop(columns=[col])
                encoding_report[original_col] = f"{col}_frequency"
        
        return data, encoding_report
    
    def _scale_features(self, data, config):
        """Scale numerical features"""
        scaled_features = []
        method = config.get('scaling_method', 'standard')
        features_to_scale = config.get('features_to_scale', [])
        
        # Filter features that actually exist in the data
        features_to_scale = [f for f in features_to_scale if f in data.columns]
        
        if not features_to_scale:
            return data, scaled_features
        
        # Initialize scaler
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            return data, scaled_features
        
        # Fit and transform
        scaled_data = scaler.fit_transform(data[features_to_scale])
        
        # Replace original features with scaled versions
        for i, feature in enumerate(features_to_scale):
            data[feature] = scaled_data[:, i]
            scaled_features.append(feature)
        
        self.scalers['numerical'] = scaler
        
        return data, scaled_features
    
    def _select_features(self, data, config):
        """Perform feature selection"""
        selection_report = {}
        
        if 'target' not in data.columns:
            return data, selection_report
        
        X = data.drop('target', axis=1)
        y = data['target']
        
        method = config.get('selection_method', 'univariate')
        
        if method == 'univariate':
            # Univariate feature selection
            score_func_name = config.get('score_func', 'f_classif')
            k_features = config.get('k_features', 10)
            
            if score_func_name == 'f_classif':
                score_func = f_classif
            else:
                score_func = mutual_info_classif
            
            selector = SelectKBest(score_func=score_func, k=min(k_features, len(X.columns)))
            X_selected = selector.fit_transform(X, y)
            
            # Get selected feature names
            selected_features = X.columns[selector.get_support()].tolist()
            feature_scores = dict(zip(X.columns, selector.scores_))
            
            # Create new dataframe with selected features
            selected_data = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
            selected_data['target'] = y
            
            selection_report['selected_features'] = selected_features
            selection_report['feature_scores'] = feature_scores
            selection_report['selection_method'] = 'univariate'
            
            self.feature_selectors['univariate'] = selector
            
            return selected_data, selection_report
        
        elif method == 'correlation':
            # Correlation-based feature selection
            threshold = config.get('correlation_threshold', 0.8)
            
            # Calculate correlation matrix
            corr_matrix = X.corr().abs()
            
            # Find highly correlated feature pairs
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if corr_matrix.iloc[i, j] > threshold:
                        high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))
            
            # Remove features with high correlation
            features_to_remove = set()
            for feat1, feat2 in high_corr_pairs:
                # Remove the feature with lower correlation to target
                if 'target' in data.columns:
                    corr1 = abs(data[feat1].corr(y))
                    corr2 = abs(data[feat2].corr(y))
                    if corr1 < corr2:
                        features_to_remove.add(feat1)
                    else:
                        features_to_remove.add(feat2)
                else:
                    features_to_remove.add(feat2)  # Remove second feature by default
            
            selected_features = [f for f in X.columns if f not in features_to_remove]
            selected_data = data[selected_features + ['target']]
            
            selection_report['selected_features'] = selected_features
            selection_report['removed_features'] = list(features_to_remove)
            selection_report['selection_method'] = 'correlation'
            
            return selected_data, selection_report
        
        elif method == 'recursive':
            # Recursive Feature Elimination
            estimator = LogisticRegression(random_state=42, max_iter=1000)
            n_features = config.get('k_features', 10)
            
            selector = RFE(estimator, n_features_to_select=min(n_features, len(X.columns)))
            X_selected = selector.fit_transform(X, y)
            
            selected_features = X.columns[selector.get_support()].tolist()
            
            selected_data = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
            selected_data['target'] = y
            
            selection_report['selected_features'] = selected_features
            selection_report['feature_ranking'] = dict(zip(X.columns, selector.ranking_))
            selection_report['selection_method'] = 'recursive'
            
            self.feature_selectors['recursive'] = selector
            
            return selected_data, selection_report
        
        return data, selection_report
    
    def transform_new_data(self, new_data):
        """Transform new data using fitted encoders and scalers"""
        transformed_data = new_data.copy()
        
        # Apply encoders
        for col, encoder in self.encoders.items():
            if col in transformed_data.columns:
                transformed_data[col] = encoder.transform(transformed_data[col].astype(str))
        
        # Apply scalers
        if 'numerical' in self.scalers:
            scaler = self.scalers['numerical']
            # Note: This requires knowing which features were scaled
            # In practice, you'd want to store this information
        
        return transformed_data
    
    def get_feature_importance_analysis(self, data, model=None):
        """Analyze feature importance and relationships"""
        analysis = {}
        
        if 'target' not in data.columns:
            return analysis
        
        X = data.drop('target', axis=1)
        y = data['target']
        
        # Correlation with target
        correlations = {}
        for col in X.select_dtypes(include=[np.number]).columns:
            correlations[col] = abs(X[col].corr(y))
        
        analysis['target_correlations'] = dict(sorted(correlations.items(), 
                                                    key=lambda x: x[1], reverse=True))
        
        # Feature variance
        variances = {}
        for col in X.select_dtypes(include=[np.number]).columns:
            variances[col] = X[col].var()
        
        analysis['feature_variances'] = variances
        
        # Missing value percentages
        missing_percentages = {}
        for col in X.columns:
            missing_pct = X[col].isnull().sum() / len(X) * 100
            if missing_pct > 0:
                missing_percentages[col] = missing_pct
        
        analysis['missing_percentages'] = missing_percentages
        
        return analysis
