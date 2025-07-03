import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.ensemble import IsolationForest
from scipy import stats
import re

class DataProcessor:
    """
    Comprehensive data preprocessing utility for credit scoring models.
    Handles missing values, outliers, data types, and basic transformations.
    """
    
    def __init__(self):
        self.imputers = {}
        self.scalers = {}
        self.outlier_detectors = {}
        self.processing_report = {}
    
    def preprocess_data(self, data, config):
        """
        Main preprocessing pipeline
        
        Args:
            data (pd.DataFrame): Raw data to preprocess
            config (dict): Configuration parameters for preprocessing
            
        Returns:
            tuple: (processed_data, processing_report)
        """
        processed_data = data.copy()
        report = {}
        
        # 1. Standardize column names
        if config.get('standardize_names', True):
            processed_data, renamed_cols = self._standardize_column_names(processed_data)
            if renamed_cols:
                report['columns_renamed'] = renamed_cols
        
        # 2. Remove duplicate rows
        if config.get('remove_duplicates', True):
            initial_rows = len(processed_data)
            processed_data = processed_data.drop_duplicates()
            duplicates_removed = initial_rows - len(processed_data)
            if duplicates_removed > 0:
                report['duplicates_removed'] = duplicates_removed
        
        # 3. Remove constant columns
        if config.get('remove_constant', True):
            constant_cols = self._identify_constant_columns(processed_data)
            if constant_cols:
                processed_data = processed_data.drop(columns=constant_cols)
                report['constant_columns_removed'] = constant_cols
        
        # 4. Handle missing values
        processed_data, missing_report = self._handle_missing_values(
            processed_data, config
        )
        if missing_report:
            report['missing_handled'] = missing_report
        
        # 5. Detect and handle outliers
        if config.get('outlier_method', 'none') != 'none':
            processed_data, outlier_report = self._handle_outliers(
                processed_data, config
            )
            if outlier_report:
                report['outliers_detected'] = outlier_report
                report['outlier_action'] = config.get('outlier_action', 'keep')
        
        # 6. Optimize data types
        processed_data = self._optimize_data_types(processed_data)
        
        return processed_data, report
    
    def _standardize_column_names(self, data):
        """Standardize column names to lowercase with underscores"""
        renamed_cols = {}
        new_columns = {}
        
        for col in data.columns:
            # Convert to lowercase and replace spaces/special chars with underscores
            new_name = re.sub(r'[^a-zA-Z0-9_]', '_', col.lower())
            new_name = re.sub(r'_+', '_', new_name)  # Remove multiple underscores
            new_name = new_name.strip('_')  # Remove leading/trailing underscores
            
            if new_name != col:
                renamed_cols[col] = new_name
                new_columns[col] = new_name
        
        if new_columns:
            data = data.rename(columns=new_columns)
        
        return data, renamed_cols
    
    def _identify_constant_columns(self, data):
        """Identify columns with constant values"""
        constant_cols = []
        for col in data.columns:
            if data[col].nunique() <= 1:
                constant_cols.append(col)
        return constant_cols
    
    def _handle_missing_values(self, data, config):
        """Handle missing values based on configuration"""
        missing_report = {}
        
        # Separate numerical and categorical columns
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        
        # Handle numerical missing values
        if len(numerical_cols) > 0:
            missing_numerical = data[numerical_cols].isnull().sum()
            missing_numerical = missing_numerical[missing_numerical > 0]
            
            if len(missing_numerical) > 0:
                strategy = config.get('missing_strategy_num', 'median')
                
                if strategy == 'drop':
                    # Drop rows with missing numerical values
                    data = data.dropna(subset=numerical_cols)
                elif strategy == 'forward_fill':
                    data[numerical_cols] = data[numerical_cols].fillna(method='ffill')
                else:
                    # Use sklearn imputer
                    imputer = SimpleImputer(strategy=strategy)
                    data[numerical_cols] = imputer.fit_transform(data[numerical_cols])
                    self.imputers['numerical'] = imputer
                
                for col in missing_numerical.index:
                    missing_report[col] = {
                        'count': missing_numerical[col],
                        'strategy': strategy
                    }
        
        # Handle categorical missing values
        if len(categorical_cols) > 0:
            missing_categorical = data[categorical_cols].isnull().sum()
            missing_categorical = missing_categorical[missing_categorical > 0]
            
            if len(missing_categorical) > 0:
                strategy = config.get('missing_strategy_cat', 'mode')
                
                if strategy == 'drop':
                    data = data.dropna(subset=categorical_cols)
                elif strategy == 'constant':
                    fill_value = config.get('missing_constant', 'Unknown')
                    data[categorical_cols] = data[categorical_cols].fillna(fill_value)
                else:
                    # Use mode
                    for col in categorical_cols:
                        if data[col].isnull().any():
                            mode_value = data[col].mode()
                            if len(mode_value) > 0:
                                data[col] = data[col].fillna(mode_value[0])
                            else:
                                data[col] = data[col].fillna('Unknown')
                
                for col in missing_categorical.index:
                    missing_report[col] = {
                        'count': missing_categorical[col],
                        'strategy': strategy
                    }
        
        return data, missing_report
    
    def _handle_outliers(self, data, config):
        """Detect and handle outliers"""
        outlier_report = {}
        method = config.get('outlier_method', 'iqr')
        action = config.get('outlier_action', 'keep')
        
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            if col == 'target':  # Skip target variable
                continue
                
            outliers = self._detect_outliers(data[col], method, config)
            
            if len(outliers) > 0:
                outlier_report[col] = len(outliers)
                
                if action == 'remove':
                    # Remove outlier rows
                    data = data[~data.index.isin(outliers)]
                elif action == 'cap':
                    # Cap outliers to threshold values
                    if method == 'iqr':
                        Q1 = data[col].quantile(0.25)
                        Q3 = data[col].quantile(0.75)
                        IQR = Q3 - Q1
                        threshold = config.get('iqr_threshold', 1.5)
                        lower_bound = Q1 - threshold * IQR
                        upper_bound = Q3 + threshold * IQR
                        
                        data[col] = np.where(data[col] < lower_bound, lower_bound, data[col])
                        data[col] = np.where(data[col] > upper_bound, upper_bound, data[col])
                    
                    elif method == 'zscore':
                        threshold = config.get('zscore_threshold', 3.0)
                        z_scores = np.abs(stats.zscore(data[col]))
                        mean_val = data[col].mean()
                        std_val = data[col].std()
                        
                        lower_bound = mean_val - threshold * std_val
                        upper_bound = mean_val + threshold * std_val
                        
                        data[col] = np.where(data[col] < lower_bound, lower_bound, data[col])
                        data[col] = np.where(data[col] > upper_bound, upper_bound, data[col])
        
        return data, outlier_report
    
    def _detect_outliers(self, series, method, config):
        """Detect outliers using specified method"""
        outliers = []
        
        if method == 'iqr':
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            threshold = config.get('iqr_threshold', 1.5)
            
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            outliers = series[(series < lower_bound) | (series > upper_bound)].index.tolist()
        
        elif method == 'zscore':
            threshold = config.get('zscore_threshold', 3.0)
            z_scores = np.abs(stats.zscore(series.dropna()))
            outliers = series.dropna().iloc[z_scores > threshold].index.tolist()
        
        elif method == 'isolation_forest':
            # Use isolation forest for multivariate outlier detection
            if len(series.dropna()) > 10:  # Need sufficient data
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                outlier_labels = iso_forest.fit_predict(series.dropna().values.reshape(-1, 1))
                outliers = series.dropna().iloc[outlier_labels == -1].index.tolist()
        
        return outliers
    
    def _optimize_data_types(self, data):
        """Optimize data types for memory efficiency"""
        for col in data.columns:
            col_type = data[col].dtype
            
            if col_type in ['int64', 'int32']:
                # Optimize integer types
                col_min = data[col].min()
                col_max = data[col].max()
                
                if col_min >= -128 and col_max <= 127:
                    data[col] = data[col].astype('int8')
                elif col_min >= -32768 and col_max <= 32767:
                    data[col] = data[col].astype('int16')
                elif col_min >= -2147483648 and col_max <= 2147483647:
                    data[col] = data[col].astype('int32')
            
            elif col_type in ['float64']:
                # Optimize float types
                data[col] = pd.to_numeric(data[col], downcast='float')
            
            elif col_type == 'object':
                # Convert to category if beneficial
                if data[col].nunique() / len(data) < 0.5:  # Less than 50% unique values
                    data[col] = data[col].astype('category')
        
        return data
    
    def get_data_quality_report(self, data):
        """Generate comprehensive data quality report"""
        report = {
            'shape': data.shape,
            'memory_usage': data.memory_usage(deep=True).sum(),
            'missing_values': data.isnull().sum().sum(),
            'duplicate_rows': data.duplicated().sum(),
            'data_types': data.dtypes.value_counts().to_dict(),
            'numerical_features': len(data.select_dtypes(include=[np.number]).columns),
            'categorical_features': len(data.select_dtypes(include=['object', 'category']).columns)
        }
        
        # Check for high cardinality categorical features
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        high_cardinality = {}
        for col in categorical_cols:
            unique_ratio = data[col].nunique() / len(data)
            if unique_ratio > 0.8:
                high_cardinality[col] = data[col].nunique()
        
        if high_cardinality:
            report['high_cardinality_features'] = high_cardinality
        
        # Check for skewed numerical features
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        skewed_features = {}
        for col in numerical_cols:
            skewness = abs(data[col].skew())
            if skewness > 2:  # Highly skewed
                skewed_features[col] = skewness
        
        if skewed_features:
            report['skewed_features'] = skewed_features
        
        return report
    
    def save_preprocessing_pipeline(self, filepath):
        """Save preprocessing components for later use"""
        import pickle
        
        pipeline_components = {
            'imputers': self.imputers,
            'scalers': self.scalers,
            'outlier_detectors': self.outlier_detectors
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(pipeline_components, f)
    
    def load_preprocessing_pipeline(self, filepath):
        """Load preprocessing components"""
        import pickle
        
        with open(filepath, 'rb') as f:
            pipeline_components = pickle.load(f)
        
        self.imputers = pipeline_components.get('imputers', {})
        self.scalers = pipeline_components.get('scalers', {})
        self.outlier_detectors = pipeline_components.get('outlier_detectors', {})
