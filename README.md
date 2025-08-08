# VolleyVision ML - Machine Learning Volleyball Analytics Repository

This repository contains a comprehensive volleyball performance analytics system built with machine learning techniques. The system uses real volleyball data to train and test various ML models including regression algorithms, classification methods, and anomaly detection techniques.
https://shreeya1-pixel.github.io/VolleyVisionML/

## Key Features

### Machine Learning Implementation
- **Performance Prediction**: Uses multiple regression algorithms (Random Forest, Gradient Boosting, Linear Regression, Ridge, Lasso, Support Vector Regression) to predict player performance scores
- **Anomaly Detection**: Implements Isolation Forest, Local Outlier Factor, Elliptic Envelope, and One-Class SVM to identify unusual performance patterns
- **Mood Classification**: Applies Random Forest, Gradient Boosting, Logistic Regression, and SVM classifiers to predict player mood and energy levels
- **Real-time Analysis**: Provides live predictions and insights as data is input

### Dataset and Training
- **Custom Volleyball Dataset**: Contains data for 150+ players with 25+ matches each, featuring realistic volleyball statistics
- **Feature Engineering**: Includes 20+ volleyball-specific features such as efficiency metrics, rolling averages, and performance trends
- **Model Training**: All models are trained on this dataset with proper train/validation/test splits
- **Performance Metrics**: Models are evaluated using MSE, MAE, R² for regression and accuracy for classification

### Analytics and Visualization
- **Performance Trends**: Time-series analysis of player performance over multiple matches
- **Skill Radar Analysis**: Multi-dimensional visualization of player skills
- **Anomaly Detection Plots**: Visual identification of unusual performance patterns
- **Player Statistics**: Detailed analytics including win rates, averages, and performance trends

### User Interface
- **Modern Web Interface**: Responsive design with dark theme and glassmorphism effects
- **Real-time Updates**: Live predictions and visualizations as data changes
- **Interactive Charts**: Multiple chart types with dynamic updates
- **Tabbed Navigation**: Organized sections for different types of analysis

## Installation and Setup

1. **Clone this repository**
   ```bash
   git clone https://github.com/Shreeya1-pixel/VolleyVisionML.git
   cd VolleyVisionML
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python app.py
   ```

4. **Access the web interface**
   - Open your browser and navigate to `http://localhost:5001`
   - The ML pipeline will automatically initialize and train models on startup

## Machine Learning Models

### Performance Predictor (Regression)
- **Algorithms**: Random Forest Regressor, Gradient Boosting Regressor, Linear Regression, Ridge Regression, Lasso Regression, Support Vector Regressor
- **Features**: Spike accuracy, blocks, digs, serves, reaction time, efficiency ratios, rolling averages, performance trends
- **Output**: Performance score prediction (0-100 scale) with confidence intervals
- **Training**: Models are trained on historical match data with cross-validation

### Anomaly Detector
- **Algorithms**: Isolation Forest, Local Outlier Factor, Elliptic Envelope, One-Class SVM
- **Features**: Performance metrics, reaction times, efficiency ratios, statistical outliers
- **Output**: Anomaly detection with severity levels and explanatory factors
- **Training**: Unsupervised learning on normal performance patterns

### Mood Predictor (Classification)
- **Algorithms**: Random Forest Classifier, Gradient Boosting Classifier, Logistic Regression, Support Vector Classifier
- **Features**: Performance correlation with mood indicators, energy levels, match outcomes
- **Output**: Mood categories (Low/Medium/High) and energy level predictions
- **Training**: Supervised learning on labeled mood data

## Usage Guide

### Basic Workflow
1. **Enter Player Information**: Input player name and basic details
2. **Input Performance Data**: Enter match statistics (spike accuracy, blocks, digs, serves, reaction time)
3. **View ML Predictions**: See real-time predictions from trained models
4. **Analyze Results**: Review performance breakdown, skill radar, and improvement suggestions

### Advanced Features
- **Autofill Sample Data**: Use pre-filled values to test the system
- **Real-time Predictions**: Watch predictions update as you modify input values
- **Player Analysis**: View historical data and performance trends
- **Anomaly Detection**: Identify unusual performance patterns using ML algorithms
- **Mood Insights**: Understand performance-mood correlations

### API Endpoints
- `POST /api/predict` - Get performance predictions from trained regression models
- `POST /api/anomaly` - Detect anomalies using trained anomaly detection models
- `POST /api/mood` - Predict mood using trained classification models
- `GET /api/player/<id>` - Retrieve player data and statistics
- `GET /api/insights` - Get dataset insights and overall statistics
- `POST /api/retrain` - Retrain models with new data

## Data and Training

### Dataset Structure
The system uses a custom-generated volleyball dataset with the following features:
- Player demographics (name, position, age, height, experience)
- Match statistics (spike accuracy, blocks, serves, digs, errors, reaction time)
- Performance metrics (efficiency ratios, rolling averages, trends)
- Match outcomes and player mood indicators

### Model Training Process
1. **Data Preprocessing**: Feature engineering, scaling, encoding categorical variables
2. **Train/Test Split**: Proper data splitting for model evaluation
3. **Hyperparameter Tuning**: Grid search for optimal model parameters
4. **Model Selection**: Automatic selection of best performing algorithm
5. **Model Persistence**: Trained models are saved for deployment

### Performance Metrics
- **Regression Models**: Mean Squared Error (MSE), Mean Absolute Error (MAE), R² Score
- **Classification Models**: Accuracy, Precision, Recall, F1-Score
- **Anomaly Detection**: Contamination ratio, outlier detection rate

## Technical Implementation

### Backend (Flask)
- RESTful API endpoints for ML predictions
- Model loading and inference
- Data preprocessing and feature engineering
- Real-time model training and updates

### Frontend (HTML/CSS/JavaScript)
- Modern responsive design
- Chart.js for data visualization
- Real-time API communication
- Dynamic content updates

### Machine Learning Stack
- **Scikit-learn**: Core ML algorithms and preprocessing
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Joblib**: Model serialization and persistence
<img width="3377" height="1969" alt="image" src="https://github.com/user-attachments/assets/56211346-7b04-4277-bea4-a8e7107db93b" />
<img width="3377" height="1951" alt="image" src="https://github.com/user-attachments/assets/acde6c60-e2fe-4051-b0e2-e3ed9e87e151" />
<img width="3388" height="1964" alt="image" src="https://github.com/user-attachments/assets/b49a2d4f-4493-4eb8-922d-4e1ba756bd63" />
<img width="3368" height="1944" alt="image" src="https://github.com/user-attachments/assets/9f23cdad-fdd8-46a8-af22-2ced06c741ef" />
<img width="3380" height="1932" alt="image" src="https://github.com/user-attachments/assets/6afe10f5-1e6c-4f9f-b6b0-b09f20748277" />





## Repository Structure

```
VolleyVisionML/
├── app.py                 # Main Flask application
├── index.html            # Web interface
├── js/main.js            # Frontend JavaScript
├── models/               # Trained ML models
│   ├── performance_predictor.py
│   ├── anomaly_detector.py
│   └── mood_predictor.py
├── scripts/              # Data processing scripts
│   ├── data_generator.py
│   └── data_preprocessing.py
├── data/                 # Dataset files
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Contributing

This repository is open for contributions. Areas for improvement include:
- Additional ML algorithms and techniques
- Enhanced feature engineering
- More comprehensive dataset
- Improved model performance
- Better visualization options

## License

This project is available under the MIT License. See the LICENSE file for details.
