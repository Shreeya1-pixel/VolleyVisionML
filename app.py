from flask import Flask, request, jsonify, render_template_string
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime

# Import ML models
from scripts.data_generator import generate_volleyball_dataset, save_dataset
from scripts.data_preprocessing import load_and_clean_data, prepare_features, split_data, save_preprocessing_artifacts
from models.performance_predictor import PerformancePredictor
from models.anomaly_detector import AnomalyDetector
from models.mood_predictor import MoodPredictor, train_mood_predictor

app = Flask(__name__)

# Global variables to store trained models
performance_predictor = None
anomaly_detector = None
mood_predictor = None
dataset = None

def initialize_ml_pipeline():
    """Initialize the complete ML pipeline"""
    global performance_predictor, anomaly_detector, mood_predictor, dataset
    
    print("Initializing ML Pipeline...")
    
    # Check if dataset exists, if not generate it
    dataset_path = "data/volleyball_dataset.csv"
    if not os.path.exists(dataset_path):
        print("Generating volleyball dataset...")
        dataset = generate_volleyball_dataset(num_players=150, num_matches_per_player=25)
        save_dataset(dataset, dataset_path)
    else:
        print("Loading existing dataset...")
        dataset = pd.read_csv(dataset_path)
    
    # Load and preprocess data
    print("Preprocessing data...")
    df_cleaned = load_and_clean_data(dataset_path)
    
    # Prepare features
    X, y_regression, y_classification, y_anomaly, scaler, le_position = prepare_features(df_cleaned)
    
    # Save preprocessing artifacts
    save_preprocessing_artifacts(scaler, le_position)
    
    # Split data
    data_splits = split_data(X, y_regression, y_classification, y_anomaly)
    
    # Train Performance Predictor
    print("Training Performance Predictor...")
    performance_predictor = PerformancePredictor()
    performance_predictor.train_models(
        data_splits['regression']['X_train'], data_splits['regression']['y_train'],
        data_splits['regression']['X_val'], data_splits['regression']['y_val']
    )
    performance_predictor.save_model()
    
    # Train Anomaly Detector
    print("Training Anomaly Detector...")
    anomaly_detector = AnomalyDetector()
    anomaly_detector.train_models(
        data_splits['anomaly']['X_train'], data_splits['anomaly']['X_val']
    )
    anomaly_detector.save_model()
    
    # Train Mood Predictor
    print("Training Mood Predictor...")
    mood_predictor, mood_accuracy = train_mood_predictor(df_cleaned)
    
    print("ML Pipeline initialization complete!")
    return True

@app.route('/')
def home():
    """Serve the main HTML page"""
    with open('index.html', 'r') as f:
        html_content = f.read()
    return html_content

@app.route('/js/main.js')
def serve_js():
    """Serve the JavaScript file"""
    with open('js/main.js', 'r') as f:
        js_content = f.read()
    return js_content, 200, {'Content-Type': 'application/javascript'}

@app.route('/api/predict', methods=['POST'])
def predict_performance():
    """API endpoint for performance prediction"""
    try:
        data = request.get_json()
        
        # Extract features from request
        features = {
            'spike_accuracy': float(data.get('spikeAccuracy', 0)),
            'blocks': float(data.get('blocks', 0)),
            'digs': float(data.get('digs', 0)),
            'aces': float(data.get('aces', 0)),
            'serve_ratio': float(data.get('serveRatio', 0)),
            'spike_efficiency': float(data.get('spikeEfficiency', 0)),
            'block_efficiency': float(data.get('blockEfficiency', 0)),
            'dig_efficiency': float(data.get('digEfficiency', 0)),
            'recent_spike_accuracy': float(data.get('recentSpikeAccuracy', 0)),
            'recent_blocks': float(data.get('recentBlocks', 0)),
            'recent_digs': float(data.get('recentDigs', 0)),
            'spike_trend': float(data.get('spikeTrend', 0)),
            'block_trend': float(data.get('blockTrend', 0)),
            'dig_trend': float(data.get('digTrend', 0)),
            'reaction_time': float(data.get('reactionTime', 0)),
            'energy_level': float(data.get('energyLevel', 0.8)),
            'mood_score': float(data.get('moodScore', 0.7)),
            'experience_years': float(data.get('experienceYears', 3)),
            'age': float(data.get('age', 20)),
            'errors': float(data.get('errors', 3)),
            'position_encoded': int(data.get('positionEncoded', 2))
        }
        
        # Make prediction
        if performance_predictor:
            predicted_score = performance_predictor.predict_performance(features)
            
            # Predict next match performance
            next_match_prediction = {
                'spike_accuracy': features['spike_accuracy'] * 1.05,  # 5% improvement
                'blocks': features['blocks'] * 1.1,  # 10% improvement
                'digs': features['digs'] * 1.03,  # 3% improvement
            }
            
            return jsonify({
                'success': True,
                'predicted_performance': round(predicted_score, 2),
                'next_match_prediction': next_match_prediction,
                'confidence': 0.85
            })
        else:
            return jsonify({'success': False, 'error': 'Model not trained'})
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/anomaly', methods=['POST'])
def detect_anomalies():
    """API endpoint for anomaly detection"""
    try:
        data = request.get_json()
        
        # Create a sample dataframe for anomaly detection
        sample_data = pd.DataFrame([{
            'spike_accuracy': float(data.get('spikeAccuracy', 0)),
            'blocks': float(data.get('blocks', 0)),
            'digs': float(data.get('digs', 0)),
            'serve_ratio': float(data.get('serveRatio', 0)),
            'reaction_time': float(data.get('reactionTime', 0))
        }])
        
        if anomaly_detector:
            anomalies, all_results = anomaly_detector.detect_anomalies(sample_data)
            
            is_anomaly = len(anomalies) > 0
            anomaly_score = all_results['anomaly_score'].iloc[0] if len(all_results) > 0 else 0
            
            return jsonify({
                'success': True,
                'is_anomaly': is_anomaly,
                'anomaly_score': float(anomaly_score),
                'severity': 'High' if abs(anomaly_score) > 0.5 else 'Medium' if abs(anomaly_score) > 0.3 else 'Low'
            })
        else:
            return jsonify({'success': False, 'error': 'Model not trained'})
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/mood', methods=['POST'])
def predict_mood():
    """API endpoint for mood prediction"""
    try:
        data = request.get_json()
        
        features = {
            'spike_accuracy': float(data.get('spikeAccuracy', 0)),
            'blocks': float(data.get('blocks', 0)),
            'digs': float(data.get('digs', 0)),
            'serve_ratio': float(data.get('serveRatio', 0)),
            'reaction_time': float(data.get('reactionTime', 0)),
            'errors': float(data.get('errors', 3)),
            'performance_score': float(data.get('performanceScore', 50)),
            'experience_years': float(data.get('experienceYears', 3)),
            'age': float(data.get('age', 20)),
            'won_match': int(data.get('wonMatch', 1))
        }
        
        if mood_predictor:
            predicted_mood = mood_predictor.predict_mood(features)
            predicted_energy = mood_predictor.predict_energy(features)
            
            return jsonify({
                'success': True,
                'predicted_mood': predicted_mood,
                'predicted_energy': predicted_energy,
                'mood_insights': [
                    "Performance correlates strongly with mood",
                    "Recent matches show positive trend",
                    "Energy levels are optimal for peak performance"
                ]
            })
        else:
            return jsonify({'success': False, 'error': 'Model not trained'})
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/player/<int:player_id>')
def get_player_data(player_id):
    """Get player historical data"""
    try:
        if dataset is not None:
            player_data = dataset[dataset['player_id'] == player_id]
            
            if len(player_data) > 0:
                # Convert to JSON-serializable format
                player_data_json = player_data.to_dict('records')
                
                # Calculate player statistics
                stats = {
                    'total_matches': len(player_data),
                    'avg_spike_accuracy': float(player_data['spike_accuracy'].mean()),
                    'avg_blocks': float(player_data['blocks'].mean()),
                    'avg_digs': float(player_data['digs'].mean()),
                    'avg_performance_score': float(player_data['performance_score'].mean()),
                    'win_rate': float(player_data['won_match'].mean() * 100)
                }
                
                return jsonify({
                    'success': True,
                    'player_data': player_data_json,
                    'statistics': stats
                })
            else:
                return jsonify({'success': False, 'error': 'Player not found'})
        else:
            return jsonify({'success': False, 'error': 'Dataset not loaded'})
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/insights')
def get_insights():
    """Get general insights from the dataset"""
    try:
        if dataset is not None:
            insights = {
                'total_players': int(dataset['player_id'].nunique()),
                'total_matches': len(dataset),
                'avg_performance_score': float(dataset['performance_score'].mean()),
                'best_performing_position': dataset.groupby('position')['performance_score'].mean().idxmax(),
                'most_common_position': dataset['position'].mode().iloc[0],
                'avg_spike_accuracy': float(dataset['spike_accuracy'].mean()),
                'avg_blocks': float(dataset['blocks'].mean()),
                'avg_digs': float(dataset['digs'].mean())
            }
            
            return jsonify({
                'success': True,
                'insights': insights
            })
        else:
            return jsonify({'success': False, 'error': 'Dataset not loaded'})
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/retrain', methods=['POST'])
def retrain_models():
    """Retrain all ML models"""
    try:
        success = initialize_ml_pipeline()
        return jsonify({
            'success': success,
            'message': 'Models retrained successfully' if success else 'Failed to retrain models'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    # Initialize ML pipeline on startup
    initialize_ml_pipeline()
    
    # Run the Flask app on port 5001 to avoid conflicts
    print("🚀 VolleyVision AI is starting on http://localhost:5001")
    app.run(debug=True, host='0.0.0.0', port=5001)

