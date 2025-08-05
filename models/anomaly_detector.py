import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import pickle

class AnomalyDetector:
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.scaler = StandardScaler()
        self.anomaly_scores = None
        
    def train_models(self, X_train, X_val, contamination=0.1):
        """Train multiple anomaly detection models"""
        
        # Scale the data
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Define models to try
        models = {
            'isolation_forest': IsolationForest(
                contamination=contamination, 
                random_state=42,
                n_estimators=100
            ),
            'local_outlier_factor': LocalOutlierFactor(
                contamination=contamination,
                n_neighbors=20,
                novelty=True
            ),
            'elliptic_envelope': EllipticEnvelope(
                contamination=contamination,
                random_state=42
            ),
            'one_class_svm': OneClassSVM(
                kernel='rbf',
                nu=contamination,
                gamma='scale'
            )
        }
        
        best_score = 0
        
        for name, model in models.items():
            print(f"Training {name}...")
            
            # Train the model
            if name == 'local_outlier_factor':
                model.fit(X_train_scaled)
                # For LOF, we need to use predict method differently
                y_pred = model.predict(X_val_scaled)
            else:
                model.fit(X_train_scaled)
                y_pred = model.predict(X_val_scaled)
            
            # Calculate anomaly score
            if hasattr(model, 'score_samples'):
                scores = model.score_samples(X_val_scaled)
            elif hasattr(model, 'decision_function'):
                scores = model.decision_function(X_val_scaled)
            else:
                scores = np.zeros(len(X_val_scaled))
            
            # Store model
            self.models[name] = {
                'model': model,
                'predictions': y_pred,
                'scores': scores
            }
            
            # For now, we'll use the model that detects the most anomalies
            # In a real scenario, you'd want to validate with labeled data
            anomaly_count = np.sum(y_pred == -1)
            if anomaly_count > best_score:
                best_score = anomaly_count
                self.best_model = model
                self.best_model_name = name
        
        print(f"\nBest model: {self.best_model_name}")
        print(f"Anomalies detected: {best_score}")
        
        return self.best_model
    
    def detect_anomalies(self, data, threshold=None):
        """Detect anomalies in the given data"""
        if self.best_model is None:
            raise ValueError("Model not trained yet. Call train_models() first.")
        
        # Scale the data
        data_scaled = self.scaler.transform(data)
        
        # Make predictions
        predictions = self.best_model.predict(data_scaled)
        
        # Get anomaly scores
        if hasattr(self.best_model, 'score_samples'):
            scores = self.best_model.score_samples(data_scaled)
        elif hasattr(self.best_model, 'decision_function'):
            scores = self.best_model.decision_function(data_scaled)
        else:
            scores = np.zeros(len(data))
        
        # Create results dataframe
        results = data.copy()
        results['anomaly'] = predictions
        results['anomaly_score'] = scores
        
        # Filter anomalies
        anomalies = results[results['anomaly'] == -1]
        
        return anomalies, results
    
    def detect_performance_anomalies(self, df, features=['spike_accuracy', 'blocks', 'digs', 'serve_ratio', 'reaction_time']):
        """Detect performance anomalies in volleyball data"""
        
        # Prepare features for anomaly detection
        X = df[features].copy()
        
        # Remove any NaN values
        X = X.dropna()
        
        if len(X) == 0:
            print("No valid data for anomaly detection")
            return pd.DataFrame()
        
        # Train models if not already trained
        if self.best_model is None:
            # Split data for training
            train_size = int(0.8 * len(X))
            X_train = X[:train_size]
            X_val = X[train_size:]
            
            self.train_models(X_train, X_val)
        
        # Detect anomalies
        anomalies, all_results = self.detect_anomalies(X)
        
        # Add player information back to results
        if len(anomalies) > 0:
            anomaly_indices = anomalies.index
            anomaly_data = df.loc[anomaly_indices].copy()
            anomaly_data['anomaly_score'] = anomalies['anomaly_score'].values
            
            # Sort by anomaly score (most anomalous first)
            anomaly_data = anomaly_data.sort_values('anomaly_score')
            
            return anomaly_data
        else:
            return pd.DataFrame()
    
    def get_anomaly_reasons(self, anomaly_data, features=['spike_accuracy', 'blocks', 'digs', 'serve_ratio', 'reaction_time']):
        """Provide reasons why a performance was flagged as anomalous"""
        
        reasons = []
        
        for idx, row in anomaly_data.iterrows():
            player_reasons = []
            
            # Check each feature for extreme values
            for feature in features:
                value = row[feature]
                
                if feature == 'spike_accuracy':
                    if value < 50:
                        player_reasons.append(f"Very low spike accuracy ({value:.1f}%)")
                    elif value > 95:
                        player_reasons.append(f"Unusually high spike accuracy ({value:.1f}%)")
                
                elif feature == 'blocks':
                    if value < 1:
                        player_reasons.append(f"Very few blocks ({value})")
                    elif value > 8:
                        player_reasons.append(f"Unusually high blocks ({value})")
                
                elif feature == 'digs':
                    if value < 5:
                        player_reasons.append(f"Very few digs ({value})")
                    elif value > 30:
                        player_reasons.append(f"Unusually high digs ({value})")
                
                elif feature == 'serve_ratio':
                    if value < 0.1:
                        player_reasons.append(f"Very low serve ratio ({value:.2f})")
                    elif value > 0.5:
                        player_reasons.append(f"Unusually high serve ratio ({value:.2f})")
                
                elif feature == 'reaction_time':
                    if value > 450:
                        player_reasons.append(f"Very slow reaction time ({value:.1f}ms)")
                    elif value < 250:
                        player_reasons.append(f"Unusually fast reaction time ({value:.1f}ms)")
            
            reasons.append({
                'player_id': row.get('player_id', idx),
                'name': row.get('name', f'Player {idx}'),
                'match_date': row.get('match_date', 'Unknown'),
                'anomaly_score': row['anomaly_score'],
                'reasons': player_reasons
            })
        
        return reasons
    
    def save_model(self, filename="models/anomaly_detector.pkl"):
        """Save the trained model"""
        if self.best_model is None:
            raise ValueError("No model to save. Train the model first.")
        
        model_data = {
            'model': self.best_model,
            'model_name': self.best_model_name,
            'scaler': self.scaler,
            'models': self.models
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Anomaly detector saved to {filename}")
    
    def load_model(self, filename="models/anomaly_detector.pkl"):
        """Load a trained model"""
        with open(filename, 'rb') as f:
            model_data = pickle.load(f)
        
        self.best_model = model_data['model']
        self.best_model_name = model_data['model_name']
        self.scaler = model_data['scaler']
        self.models = model_data['models']
        
        print(f"Anomaly detector loaded from {filename}")

def detect_anomalies(df):
    """Legacy function for backward compatibility"""
    detector = AnomalyDetector()
    anomalies = detector.detect_performance_anomalies(df)
    return anomalies

