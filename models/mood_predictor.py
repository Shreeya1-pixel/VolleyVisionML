import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle

class MoodPredictor:
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.scaler = StandardScaler()
        self.mood_encoder = LabelEncoder()
        self.energy_encoder = LabelEncoder()
        
    def prepare_mood_data(self, df):
        """Prepare data for mood prediction"""
        
        # Create mood categories based on performance and other factors
        df['mood_category'] = pd.cut(
            df['mood_score'], 
            bins=[0, 0.6, 0.8, 1.0], 
            labels=['Low', 'Medium', 'High']
        )
        
        df['energy_category'] = pd.cut(
            df['energy_level'], 
            bins=[0, 0.7, 0.85, 1.0], 
            labels=['Low', 'Medium', 'High']
        )
        
        # Features for mood prediction
        mood_features = [
            'spike_accuracy', 'blocks', 'digs', 'serve_ratio',
            'reaction_time', 'errors', 'performance_score',
            'experience_years', 'age', 'won_match'
        ]
        
        # Prepare features
        X = df[mood_features].copy()
        y_mood = df['mood_category']
        y_energy = df['energy_category']
        
        # Encode target variables
        y_mood_encoded = self.mood_encoder.fit_transform(y_mood)
        y_energy_encoded = self.energy_encoder.fit_transform(y_energy)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        
        return X_scaled, y_mood_encoded, y_energy_encoded
    
    def train_mood_models(self, X_train, y_train, X_val, y_val, target_type='mood'):
        """Train models for mood or energy prediction"""
        
        # Define models
        models = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'svm': SVC(kernel='rbf', random_state=42)
        }
        
        best_score = 0
        
        for name, model in models.items():
            print(f"Training {name} for {target_type} prediction...")
            
            # Train the model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_val)
            
            # Calculate accuracy
            accuracy = accuracy_score(y_val, y_pred)
            
            print(f"{name} - Accuracy: {accuracy:.4f}")
            
            # Store model
            self.models[f"{target_type}_{name}"] = {
                'model': model,
                'accuracy': accuracy,
                'predictions': y_pred
            }
            
            # Update best model
            if accuracy > best_score:
                best_score = accuracy
                self.best_model = model
                self.best_model_name = f"{target_type}_{name}"
        
        print(f"\nBest {target_type} model: {self.best_model_name}")
        print(f"Best accuracy: {best_score:.4f}")
        
        return self.best_model, best_score
    
    def predict_mood(self, features):
        """Predict mood category for given features"""
        if self.best_model is None:
            raise ValueError("Model not trained yet. Call train_mood_models() first.")
        
        # Ensure features are in the correct format
        if isinstance(features, dict):
            features = pd.DataFrame([features])
        elif isinstance(features, list):
            features = pd.DataFrame([features])
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Make prediction
        prediction_encoded = self.best_model.predict(features_scaled)[0]
        prediction = self.mood_encoder.inverse_transform([prediction_encoded])[0]
        
        return prediction
    
    def predict_energy(self, features):
        """Predict energy level for given features"""
        # This would use the energy model
        # For now, return a simple prediction based on performance
        if isinstance(features, dict):
            performance_score = features.get('performance_score', 50)
        else:
            performance_score = features.get('performance_score', 50) if hasattr(features, 'get') else 50
        
        if performance_score > 80:
            return 'High'
        elif performance_score > 60:
            return 'Medium'
        else:
            return 'Low'
    
    def get_mood_insights(self, player_history):
        """Get insights about player mood patterns"""
        if len(player_history) == 0:
            return "No historical data available for mood analysis."
        
        # Analyze recent mood trends
        recent_matches = player_history.tail(10)
        
        avg_mood = recent_matches['mood_score'].mean()
        mood_trend = recent_matches['mood_score'].diff().mean()
        
        insights = []
        
        if avg_mood > 0.8:
            insights.append("Player shows consistently high mood levels")
        elif avg_mood < 0.6:
            insights.append("Player shows consistently low mood levels")
        else:
            insights.append("Player shows moderate mood levels")
        
        if mood_trend > 0.02:
            insights.append("Mood is improving over recent matches")
        elif mood_trend < -0.02:
            insights.append("Mood is declining over recent matches")
        else:
            insights.append("Mood is relatively stable")
        
        # Performance correlation
        mood_performance_corr = recent_matches['mood_score'].corr(recent_matches['performance_score'])
        if mood_performance_corr > 0.3:
            insights.append("Strong positive correlation between mood and performance")
        elif mood_performance_corr < -0.3:
            insights.append("Negative correlation between mood and performance")
        else:
            insights.append("Moderate correlation between mood and performance")
        
        return insights
    
    def save_model(self, filename_prefix="models/mood_predictor"):
        """Save the trained model"""
        if self.best_model is None:
            raise ValueError("No model to save. Train the model first.")
        
        model_data = {
            'model': self.best_model,
            'model_name': self.best_model_name,
            'scaler': self.scaler,
            'mood_encoder': self.mood_encoder,
            'energy_encoder': self.energy_encoder,
            'models': self.models
        }
        
        with open(f"{filename_prefix}.pkl", 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Mood predictor saved to {filename_prefix}.pkl")
    
    def load_model(self, filename_prefix="models/mood_predictor"):
        """Load a trained model"""
        with open(f"{filename_prefix}.pkl", 'rb') as f:
            model_data = pickle.load(f)
        
        self.best_model = model_data['model']
        self.best_model_name = model_data['model_name']
        self.scaler = model_data['scaler']
        self.mood_encoder = model_data['mood_encoder']
        self.energy_encoder = model_data['energy_encoder']
        self.models = model_data['models']
        
        print(f"Mood predictor loaded from {filename_prefix}.pkl")

def train_mood_predictor(df):
    """Train mood prediction models"""
    from scripts.data_preprocessing import split_data
    
    predictor = MoodPredictor()
    
    # Prepare data
    X, y_mood, y_energy = predictor.prepare_mood_data(df)
    
    # Split data
    X_train, X_temp, y_mood_train, y_mood_temp = train_test_split(
        X, y_mood, test_size=0.3, random_state=42
    )
    X_val, X_test, y_mood_val, y_mood_test = train_test_split(
        X_temp, y_mood_temp, test_size=0.5, random_state=42
    )
    
    # Train mood prediction model
    model, accuracy = predictor.train_mood_models(
        X_train, y_mood_train, X_val, y_mood_val, 'mood'
    )
    
    # Save model
    predictor.save_model()
    
    return predictor, accuracy 