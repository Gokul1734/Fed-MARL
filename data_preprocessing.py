"""
Data Preprocessing and Feature Engineering Module
Handles data cleaning, feature extraction, and preparation for ML models
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """
    Comprehensive data preprocessing pipeline for agricultural data
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.imputer = SimpleImputer(strategy='median')
        self.feature_names = None
        self.target_names = None
        
    def load_data(self, filepath):
        """Load agricultural dataset from CSV"""
        print(f" Loading data from {filepath}...")
        df = pd.read_csv(filepath)
        print(f" Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    
    def handle_missing_values(self, df):
        """Handle missing values using imputation"""
        print(" Handling missing values...")
        missing_before = df.isnull().sum().sum()
        
        # Separate numeric and non-numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if missing_before > 0:
            df[numeric_cols] = self.imputer.fit_transform(df[numeric_cols])
            print(f" Imputed {missing_before} missing values")
        else:
            print(" No missing values found")
        
        return df
    
    def remove_outliers(self, df, columns=None, threshold=3):
        """Remove outliers using Z-score method"""
        print(" Removing outliers...")
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns
        
        initial_rows = len(df)
        
        # Calculate Z-scores
        z_scores = np.abs((df[columns] - df[columns].mean()) / df[columns].std())
        
        # Remove rows where any column has Z-score > threshold
        df_clean = df[(z_scores < threshold).all(axis=1)]
        
        removed_rows = initial_rows - len(df_clean)
        print(f" Removed {removed_rows} outlier rows ({removed_rows/initial_rows*100:.2f}%)")
        
        return df_clean
    
    def engineer_features(self, df):
        """
        Create advanced features from existing data
        - Interaction terms
        - Ratios
        - Polynomial features
        - Seasonal indicators
        """
        print(" Engineering features...")
        
        df_engineered = df.copy()
        
        # NPK ratio features
        if all(col in df.columns for col in ['N', 'P', 'K']):
            df_engineered['NPK_sum'] = df['N'] + df['P'] + df['K']
            df_engineered['NPK_ratio'] = df['N'] / (df['P'] + df['K'] + 1)
            df_engineered['NP_ratio'] = df['N'] / (df['P'] + 1)
            df_engineered['NK_ratio'] = df['N'] / (df['K'] + 1)
            df_engineered['PK_ratio'] = df['P'] / (df['K'] + 1)
        
        # Soil-weather interaction terms
        if all(col in df.columns for col in ['temperature', 'humidity']):
            df_engineered['temp_humidity_interaction'] = df['temperature'] * df['humidity']
            df_engineered['heat_index'] = df['temperature'] + (df['humidity'] / 100) * 5
        
        # Moisture stress indicator
        if all(col in df.columns for col in ['rainfall', 'humidity']):
            df_engineered['moisture_stress'] = df['rainfall'] / (df['humidity'] + 1)
        
        # pH categories
        if 'ph' in df.columns:
            df_engineered['ph_acidic'] = (df['ph'] < 6.5).astype(int)
            df_engineered['ph_neutral'] = ((df['ph'] >= 6.5) & (df['ph'] <= 7.5)).astype(int)
            df_engineered['ph_alkaline'] = (df['ph'] > 7.5).astype(int)
        
        # Temperature zones
        if 'temperature' in df.columns:
            df_engineered['temp_low'] = (df['temperature'] < 20).astype(int)
            df_engineered['temp_medium'] = ((df['temperature'] >= 20) & (df['temperature'] <= 30)).astype(int)
            df_engineered['temp_high'] = (df['temperature'] > 30).astype(int)
        
        # Rainfall categories
        if 'rainfall' in df.columns:
            df_engineered['rainfall_low'] = (df['rainfall'] < 100).astype(int)
            df_engineered['rainfall_medium'] = ((df['rainfall'] >= 100) & (df['rainfall'] <= 200)).astype(int)
            df_engineered['rainfall_high'] = (df['rainfall'] > 200).astype(int)
        
        new_features = len(df_engineered.columns) - len(df.columns)
        print(f" Created {new_features} new features")
        
        return df_engineered
    
    def prepare_crop_recommendation_data(self, df, test_size=0.15, val_size=0.15, scale=True):
        """
        Prepare data for crop recommendation task
        Returns: X_train, X_val, X_test, y_train, y_val, y_test
        """
        print("\n Preparing data for Crop Recommendation...")
        
        # Separate features and target
        if 'label' in df.columns:
            X = df.drop('label', axis=1)
            y = df['label']
        else:
            raise ValueError("Target column 'label' not found")
        
        # Encode target labels
        y_encoded = self.label_encoder.fit_transform(y)
        self.target_names = self.label_encoder.classes_
        
        print(f" Classes: {list(self.target_names)}")
        print(f" Number of classes: {len(self.target_names)}")
        
        # Split data: train/val/test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
        )
        
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=42, stratify=y_temp
        )
        
        print(f" Train set: {X_train.shape[0]} samples")
        print(f" Validation set: {X_val.shape[0]} samples")
        print(f" Test set: {X_test.shape[0]} samples")
        
        # Scale features
        if scale:
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)
            X_test_scaled = self.scaler.transform(X_test)
            
            self.feature_names = X.columns.tolist()
            
            return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test
        else:
            return X_train, X_val, X_test, y_train, y_val, y_test
    
    def prepare_yield_prediction_data(self, df, target_col='yield', test_size=0.15, val_size=0.15):
        """
        Prepare data for crop yield prediction task
        """
        print("\n Preparing data for Yield Prediction...")
        
        if target_col not in df.columns:
            # If yield column doesn't exist, create synthetic yield data based on features
            print(f" Target column '{target_col}' not found. Creating synthetic yield data...")
            df = self._generate_synthetic_yield(df)
        
        # Exclude target column and label column (if present)
        columns_to_drop = [target_col]
        if 'label' in df.columns:
            columns_to_drop.append('label')
        
        X = df.drop(columns_to_drop, axis=1)
        y = df[target_col]
        
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=42
        )
        
        # Scale features (use fresh scaler for yield prediction)
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        self.scaler = scaler  # Save the scaler
        
        self.feature_names = X.columns.tolist()
        
        print(f" Train set: {X_train.shape[0]} samples")
        print(f" Validation set: {X_val.shape[0]} samples")
        print(f" Test set: {X_test.shape[0]} samples")
        
        return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test
    
    def _generate_synthetic_yield(self, df):
        """Generate synthetic yield data based on features"""
        df = df.copy()
        
        # Create yield based on NPK, rainfall, and other factors
        base_yield = 50  # Base yield in quintals per hectare
        
        if 'N' in df.columns:
            df['yield'] = base_yield + (df['N'] * 0.3) + (df['P'] * 0.25) + (df['K'] * 0.2)
        
        if 'rainfall' in df.columns:
            df['yield'] += (df['rainfall'] / 10)
        
        if 'temperature' in df.columns:
            # Optimal temperature around 25 C
            df['yield'] += 10 - abs(df['temperature'] - 25) * 0.5
        
        # Add some noise
        np.random.seed(42)
        df['yield'] += np.random.normal(0, 5, len(df))
        df['yield'] = df['yield'].clip(lower=20)  # Minimum yield
        
        return df
    
    def get_feature_statistics(self, df):
        """Get comprehensive statistics of the dataset"""
        print("\n Dataset Statistics:")
        print("="*60)
        print(df.describe())
        print("\n Data Types:")
        print(df.dtypes)
        print("\n Class Distribution (if applicable):")
        if 'label' in df.columns:
            print(df['label'].value_counts())
        
        return df.describe()


if __name__ == "__main__":
    # Test the preprocessing pipeline
    preprocessor = DataPreprocessor()
    
    # Load data
    df = preprocessor.load_data("Crop_recommendation.csv")
    
    # Preprocessing steps
    df = preprocessor.handle_missing_values(df)
    df = preprocessor.remove_outliers(df)
    df_engineered = preprocessor.engineer_features(df)
    
    # Get statistics
    preprocessor.get_feature_statistics(df_engineered)
    
    # Prepare data for crop recommendation
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.prepare_crop_recommendation_data(
        df_engineered
    )
    
    print("\n Data preprocessing complete!")
    print(f"Feature vector size: {X_train.shape[1]}")
