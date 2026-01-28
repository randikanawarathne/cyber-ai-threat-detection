"""
Data preprocessing and feature engineering for security logs
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')

class SecurityDataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_data(self, filepath):
        """Load and initial inspection of security logs"""
        df = pd.read_csv(filepath)
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"Missing values:\n{df.isnull().sum()}")
        return df
    
    def clean_data(self, df):
        """Handle missing values and outliers"""
        # Fill numerical missing values with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())
        
        # Fill categorical missing values with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df[col] = df[col].fillna(df[col].mode()[0])
        
        return df
    
    def engineer_features(self, df):
        """Create security-relevant features"""
        # Example features for network data
        if 'duration' in df.columns:
            df['log_duration'] = np.log1p(df['duration'])
        
        if 'src_bytes' in df.columns and 'dst_bytes' in df.columns:
            df['bytes_ratio'] = df['src_bytes'] / (df['dst_bytes'] + 1)
            df['total_bytes'] = df['src_bytes'] + df['dst_bytes']
        
        # Time-based features
        if 'timestamp' in df.columns:
            df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
            df['is_business_hours'] = df['hour'].between(9, 17).astype(int)
        
        return df
    
    def encode_categorical(self, df, categorical_columns):
        """Encode categorical variables"""
        for col in categorical_columns:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
        return df
    
    def select_features(self, X, y, k=20):
        """Select most important features using ANOVA F-test"""
        selector = SelectKBest(score_func=f_classif, k=min(k, X.shape[1]))
        X_selected = selector.fit_transform(X, y)
        selected_indices = selector.get_support(indices=True)
        return X_selected, selected_indices
    
    def prepare_training_data(self, df, target_column='label'):
        """Full preprocessing pipeline"""
        df = self.clean_data(df)
        df = self.engineer_features(df)
        
        # Encode categorical variables
        categorical_cols = df.select_dtypes(include=['object']).columns
        df = self.encode_categorical(df, categorical_cols)
        
        # Separate features and target
        X = df.drop(columns=[target_column] if target_column in df.columns else [])
        y = df[target_column] if target_column in df.columns else None
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y, X.columns.tolist()