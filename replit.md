# Credit Scoring Model Platform

## Overview

This is a production-ready Credit Scoring Model Platform built with Streamlit that provides a comprehensive solution for building, evaluating, and deploying credit risk assessment systems. The platform implements a complete machine learning pipeline from data upload through model deployment, specifically designed for predicting individual creditworthiness using historical financial data.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit with multi-page application structure
- **Layout**: Wide layout with sidebar navigation and expandable sections
- **State Management**: Session state for maintaining pipeline data across pages
- **Visualization**: Plotly for interactive charts and graphs

### Backend Architecture
- **Core Logic**: Modular utility classes for different ML pipeline stages
- **Data Processing**: Pandas and NumPy for data manipulation
- **ML Framework**: Scikit-learn for model training and evaluation
- **Feature Engineering**: Custom utility classes for automated feature creation
- **Model Training**: Support for multiple algorithms (Logistic Regression, Decision Trees, Random Forest, Gradient Boosting, SVM)

### Pipeline Structure
The application follows a 6-stage ML pipeline:
1. Data Upload & Initial Analysis
2. Data Preprocessing
3. Feature Engineering
4. Model Training & Hyperparameter Tuning
5. Model Evaluation & Interpretability
6. Credit Scoring & Predictions

## Key Components

### Data Processing (`utils/data_processor.py`)
- **Purpose**: Handles data cleaning, missing value imputation, outlier detection
- **Key Features**: 
  - Standardizes column names
  - Removes duplicates and constant columns
  - Multiple imputation strategies (mean, median, mode, forward fill)
  - Outlier detection using Isolation Forest and Z-score methods

### Feature Engineering (`utils/feature_engineer.py`)
- **Purpose**: Creates derived features and prepares data for ML
- **Key Features**:
  - Automated financial ratio creation (debt-to-income, payment ratios)
  - Multiple encoding strategies (Label, One-Hot)
  - Feature scaling (Standard, MinMax, Robust)
  - Feature selection using statistical methods

### Model Training (`utils/model_trainer.py`)
- **Purpose**: Trains multiple classification models with hyperparameter tuning
- **Supported Models**: Logistic Regression, Decision Trees, Random Forest, Gradient Boosting, SVM
- **Key Features**:
  - Grid search for hyperparameter optimization
  - Cross-validation for robust evaluation
  - Class weight handling for imbalanced datasets

### Model Evaluation (`utils/model_evaluator.py`)
- **Purpose**: Comprehensive model performance analysis
- **Key Features**:
  - Standard classification metrics (precision, recall, F1, ROC-AUC)
  - Confusion matrix analysis
  - Calibration curve analysis
  - Business impact metrics

### Bias Detection (`utils/bias_detector.py`)
- **Purpose**: Ensures regulatory compliance and fairness
- **Key Features**:
  - Demographic parity analysis
  - Equalized odds evaluation
  - Treatment equality assessment
  - Protected attribute bias detection

## Data Flow

1. **Data Upload**: Users upload CSV files containing credit-related features
2. **Preprocessing**: Data is cleaned, missing values handled, outliers detected
3. **Feature Engineering**: New features created, categorical variables encoded, features scaled and selected
4. **Model Training**: Multiple models trained with hyperparameter tuning and cross-validation
5. **Evaluation**: Models evaluated using comprehensive metrics and bias analysis
6. **Scoring**: Best model used for real-time credit scoring and risk assessment

### Session State Management
- `data`: Original uploaded dataset
- `processed_data`: Cleaned and preprocessed dataset
- `engineered_data`: Dataset with engineered features
- `trained_models`: Dictionary of trained models
- `model_results`: Performance metrics for each model
- `best_model`: Identifier for the best performing model

## External Dependencies

### Core Libraries
- **Streamlit**: Web application framework
- **Pandas/NumPy**: Data manipulation and numerical computing
- **Scikit-learn**: Machine learning algorithms and utilities
- **Plotly**: Interactive visualization
- **SHAP**: Model interpretability (referenced but not fully implemented)

### Machine Learning Stack
- **Classification Algorithms**: Logistic Regression, Decision Trees, Random Forest, Gradient Boosting, SVM
- **Preprocessing**: StandardScaler, RobustScaler, MinMaxScaler, SimpleImputer
- **Feature Selection**: SelectKBest, RFE, mutual information
- **Evaluation**: Comprehensive metrics suite from sklearn.metrics

## Deployment Strategy

### Current Setup
- **Platform**: Designed for Replit deployment
- **Architecture**: Single-container Streamlit application
- **State Management**: Session-based (non-persistent)
- **File Handling**: In-memory processing of uploaded CSV files

### Production Considerations
- **Scalability**: Modular design allows for easy horizontal scaling
- **Model Persistence**: Models can be saved/loaded using pickle
- **Data Storage**: Currently file-based, can be extended to database storage
- **API Integration**: Structure supports REST API wrapper development
- **Monitoring**: Framework in place for bias detection and model performance tracking

### Recommended Enhancements for Production
- Database integration for persistent data storage
- Model versioning and A/B testing capabilities
- Real-time monitoring and alerting
- Automated retraining pipelines
- Enhanced security for sensitive financial data

## Changelog
- July 03, 2025. Initial setup

## User Preferences

Preferred communication style: Simple, everyday language.