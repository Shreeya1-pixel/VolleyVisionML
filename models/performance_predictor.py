import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV, cross_val_score
import joblib
import pickle

class PerformancePredictor:
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.scaler = None
        self.feature_importance = None
        
    def train_models(self, X_train, y_train, X_val, y_val):
        """Train multiple regression models and select the best one"""
        
        # Define models to try
        models = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'linear_regression': LinearRegression(),
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=0.1),
            'svr': SVR(kernel='rbf', C=1.0, gamma='scale')
        }
        
        best_score = float('inf')
        
        for name, model in models.items():
            print(f"Training {name}...")
            
            # Train the model
            model.fit(X_train, y_train)
            
            # Make predictions on validation set
            y_pred = model.predict(X_val)
            
            # Calculate metrics
            mse = mean_squared_error(y_val, y_pred)
            mae = mean_absolute_error(y_val, y_pred)
            r2 = r2_score(y_val, y_pred)
            
            print(f"{name} - MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
            
            # Store model
            self.models[name] = {
                'model': model,
                'mse': mse,
                'mae': mae,
                'r2': r2
            }
            
            # Update best model
            if mse < best_score:
                best_score = mse
                self.best_model = model
                self.best_model_name = name
        
        print(f"\nBest model: {self.best_model_name}")
        print(f"Best MSE: {best_score:.4f}")
        
        # Get feature importance for the best model
        if hasattr(self.best_model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': self.best_model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        return self.best_model, best_score
    
    def hyperparameter_tuning(self, X_train, y_train, model_name='random_forest'):
        """Perform hyperparameter tuning for the best model"""
        
        if model_name == 'random_forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            model = RandomForestRegressor(random_state=42)
            
        elif model_name == 'gradient_boosting':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0]
            }
            model = GradientBoostingRegressor(random_state=42)
            
        else:
            print(f"Hyperparameter tuning not implemented for {model_name}")
            return self.best_model
        
        print(f"Performing hyperparameter tuning for {model_name}...")
        
        grid_search = GridSearchCV(
            model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {-grid_search.best_score_:.4f}")
        
        self.best_model = grid_search.best_estimator_
        return self.best_model
    
    def predict_performance(self, features):
        """Predict performance score for given features"""
        if self.best_model is None:
            raise ValueError("Model not trained yet. Call train_models() first.")
        
        # Ensure features are in the correct format
        if isinstance(features, dict):
            features = pd.DataFrame([features])
        elif isinstance(features, list):
            features = pd.DataFrame([features])
        
        # Make prediction
        prediction = self.best_model.predict(features)[0]
        return max(0, min(100, prediction))  # Clamp between 0 and 100
    
    def predict_next_match(self, player_history, current_stats):
        """Predict performance for the next match based on player history and current stats"""
        
        # Combine historical trends with current stats
        features = current_stats.copy()
        
        # Add trend features from history
        if len(player_history) > 0:
            recent_matches = player_history.tail(5)
            features['recent_spike_accuracy'] = recent_matches['spike_accuracy'].mean()
            features['recent_blocks'] = recent_matches['blocks'].mean()
            features['recent_digs'] = recent_matches['digs'].mean()
            features['spike_trend'] = recent_matches['spike_accuracy'].diff().mean()
            features['block_trend'] = recent_matches['blocks'].diff().mean()
            features['dig_trend'] = recent_matches['digs'].diff().mean()
        else:
            features['recent_spike_accuracy'] = features['spike_accuracy']
            features['recent_blocks'] = features['blocks']
            features['recent_digs'] = features['digs']
            features['spike_trend'] = 0
            features['block_trend'] = 0
            features['dig_trend'] = 0
        
        return self.predict_performance(features)
    
    def get_feature_importance(self):
        """Get feature importance from the best model"""
        return self.feature_importance
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate the best model on test data"""
        if self.best_model is None:
            raise ValueError("Model not trained yet. Call train_models() first.")
        
        y_pred = self.best_model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Test Results:")
        print(f"MSE: {mse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R²: {r2:.4f}")
        
        return {
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'predictions': y_pred
        }
    
    def save_model(self, filename="models/performance_predictor.pkl"):
        """Save the trained model"""
        if self.best_model is None:
            raise ValueError("No model to save. Train the model first.")
        
        model_data = {
            'model': self.best_model,
            'model_name': self.best_model_name,
            'feature_importance': self.feature_importance,
            'models': self.models
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filename}")
    
    def load_model(self, filename="models/performance_predictor.pkl"):
        """Load a trained model"""
        with open(filename, 'rb') as f:
            model_data = pickle.load(f)
        
        self.best_model = model_data['model']
        self.best_model_name = model_data['model_name']
        self.feature_importance = model_data['feature_importance']
        self.models = model_data['models']
        
        print(f"Model loaded from {filename}")

def train_model(df):
    """Legacy function for backward compatibility"""
    from scripts.data_preprocessing import prepare_features, split_data
    
    # Prepare features
    X, y_regression, _, _, _, _ = prepare_features(df)
    
    # Split data
    data_splits = split_data(X, y_regression, y_regression, y_regression)
    reg_data = data_splits['regression']
    
    # Train model
    predictor = PerformancePredictor()
    model, mse = predictor.train_models(
        reg_data['X_train'], reg_data['y_train'],
        reg_data['X_val'], reg_data['y_val']
    )
    
    # Save model
    predictor.save_model()

    return model, mse

