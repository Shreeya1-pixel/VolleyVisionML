import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pickle

def load_and_clean_data(path):
    """Load and clean the volleyball dataset"""
    df = pd.read_csv(path)
    
    # Convert date column
    df['match_date'] = pd.to_datetime(df['match_date'])
    
    # Extract serve ratio from serves column
    df[['aces', 'serve_attempts']] = df['serves'].str.split('/', expand=True).astype(int)
    df['serve_ratio'] = df['aces'] / df['serve_attempts']
    
    # Create additional features
    df['spike_efficiency'] = df['spike_success'] / df['spike_attempts']
    df['block_efficiency'] = df['blocks'] / df['block_attempts']
    df['dig_efficiency'] = df['digs'] / df['dig_attempts']
    
    # Create rolling averages for recent performance
    df = df.sort_values(['player_id', 'match_date'])
    df['recent_spike_accuracy'] = df.groupby('player_id')['spike_accuracy'].rolling(5, min_periods=1).mean().reset_index(0, drop=True)
    df['recent_blocks'] = df.groupby('player_id')['blocks'].rolling(5, min_periods=1).mean().reset_index(0, drop=True)
    df['recent_digs'] = df.groupby('player_id')['digs'].rolling(5, min_periods=1).mean().reset_index(0, drop=True)
    
    # Create performance trend features
    df['spike_trend'] = df.groupby('player_id')['spike_accuracy'].diff()
    df['block_trend'] = df.groupby('player_id')['blocks'].diff()
    df['dig_trend'] = df.groupby('player_id')['digs'].diff()
    
    # Fill NaN values
    df = df.fillna(0)
    
    # Remove outliers (performance scores beyond 3 standard deviations)
    Q1 = df['performance_score'].quantile(0.25)
    Q3 = df['performance_score'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df['performance_score'] >= lower_bound) & (df['performance_score'] <= upper_bound)]
    
    return df

def prepare_features(df):
    """Prepare features for ML models"""
    # Select relevant features
    feature_columns = [
        'spike_accuracy', 'blocks', 'digs', 'aces', 'serve_ratio',
        'spike_efficiency', 'block_efficiency', 'dig_efficiency',
        'recent_spike_accuracy', 'recent_blocks', 'recent_digs',
        'spike_trend', 'block_trend', 'dig_trend',
        'reaction_time', 'energy_level', 'mood_score',
        'experience_years', 'age', 'errors'
    ]
    
    # Encode categorical variables
    le_position = LabelEncoder()
    df['position_encoded'] = le_position.fit_transform(df['position'])
    feature_columns.append('position_encoded')
    
    # Create target variables
    df['performance_category'] = pd.cut(df['performance_score'], 
                                       bins=[0, 50, 70, 85, 100], 
                                       labels=['Poor', 'Average', 'Good', 'Excellent'])
    
    # Prepare X and y
    X = df[feature_columns].copy()
    y_regression = df['performance_score']
    y_classification = df['performance_category']
    y_anomaly = df[['spike_accuracy', 'blocks', 'digs', 'serve_ratio', 'reaction_time']]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    return X_scaled, y_regression, y_classification, y_anomaly, scaler, le_position

def split_data(X, y_regression, y_classification, y_anomaly, test_size=0.2, val_size=0.2):
    """Split data into train, validation, and test sets"""
    # Split for regression
    X_train_reg, X_temp_reg, y_train_reg, y_temp_reg = train_test_split(
        X, y_regression, test_size=test_size + val_size, random_state=42
    )
    X_val_reg, X_test_reg, y_val_reg, y_test_reg = train_test_split(
        X_temp_reg, y_temp_reg, test_size=val_size/(test_size + val_size), random_state=42
    )
    
    # Split for classification
    X_train_clf, X_temp_clf, y_train_clf, y_temp_clf = train_test_split(
        X, y_classification, test_size=test_size + val_size, random_state=42
    )
    X_val_clf, X_test_clf, y_val_clf, y_test_clf = train_test_split(
        X_temp_clf, y_temp_clf, test_size=val_size/(test_size + val_size), random_state=42
    )
    
    # Split for anomaly detection
    X_train_anom, X_temp_anom, y_train_anom, y_temp_anom = train_test_split(
        X, y_anomaly, test_size=test_size + val_size, random_state=42
    )
    X_val_anom, X_test_anom, y_val_anom, y_test_anom = train_test_split(
        X_temp_anom, y_temp_anom, test_size=val_size/(test_size + val_size), random_state=42
    )
    
    return {
        'regression': {
            'X_train': X_train_reg, 'X_val': X_val_reg, 'X_test': X_test_reg,
            'y_train': y_train_reg, 'y_val': y_val_reg, 'y_test': y_test_reg
        },
        'classification': {
            'X_train': X_train_clf, 'X_val': X_val_clf, 'X_test': X_test_clf,
            'y_train': y_train_clf, 'y_val': y_val_clf, 'y_test': y_test_clf
        },
        'anomaly': {
            'X_train': X_train_anom, 'X_val': X_val_anom, 'X_test': X_test_anom,
            'y_train': y_train_anom, 'y_val': y_val_anom, 'y_test': y_test_anom
        }
    }

def save_preprocessing_artifacts(scaler, le_position, filename_prefix="models/preprocessing"):
    """Save preprocessing artifacts for later use"""
    with open(f"{filename_prefix}_scaler.pkl", 'wb') as f:
        pickle.dump(scaler, f)
    
    with open(f"{filename_prefix}_label_encoder.pkl", 'wb') as f:
        pickle.dump(le_position, f)

def load_preprocessing_artifacts(filename_prefix="models/preprocessing"):
    """Load preprocessing artifacts"""
    with open(f"{filename_prefix}_scaler.pkl", 'rb') as f:
        scaler = pickle.load(f)
    
    with open(f"{filename_prefix}_label_encoder.pkl", 'rb') as f:
        le_position = pickle.load(f)
    
    return scaler, le_position

