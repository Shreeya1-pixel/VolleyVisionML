# VolleyVision AI - ML-Powered Volleyball Performance Tracker

VolleyVision AI is a comprehensive volleyball performance tracking system powered by machine learning. It provides real-time performance predictions, anomaly detection, mood analysis, and advanced visualizations for volleyball players and coaches.

## 🚀 Features

### Core ML Capabilities
- **Performance Prediction**: Advanced ML models predict player performance scores using multiple algorithms (Random Forest, Gradient Boosting, Linear Regression, SVM)
- **Anomaly Detection**: Identifies unusual performance patterns using Isolation Forest, Local Outlier Factor, and other anomaly detection algorithms
- **Mood Prediction**: Predicts player mood and energy levels based on performance metrics
- **Real-time Analysis**: Live predictions and insights as you input data

### Data & Analytics
- **Comprehensive Dataset**: 150+ players with 25+ matches each, featuring realistic volleyball statistics
- **Advanced Visualizations**: Performance trends, radar charts, anomaly detection plots
- **Player Statistics**: Detailed analytics including win rates, averages, and performance trends
- **Dataset Insights**: Overall statistics and patterns from the volleyball dataset

### User Interface
- **Modern Design**: Beautiful, responsive interface with dark theme
- **Real-time Updates**: Live predictions and visualizations
- **ML-Powered Autofill**: AI-suggested values based on historical data
- **Interactive Charts**: Multiple chart types with real-time updates

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd volleyvision-ai
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python app.py
   ```

4. **Access the application**
   - Open your browser and go to `http://localhost:5000`
   - The ML pipeline will automatically initialize on startup

## 📊 ML Models

### Performance Predictor
- **Algorithms**: Random Forest, Gradient Boosting, Linear Regression, Ridge, Lasso, SVR
- **Features**: 20+ volleyball-specific features including efficiency metrics, trends, and player characteristics
- **Output**: Performance score prediction (0-100) with confidence levels

### Anomaly Detector
- **Algorithms**: Isolation Forest, Local Outlier Factor, Elliptic Envelope, One-Class SVM
- **Features**: Performance metrics, reaction times, efficiency ratios
- **Output**: Anomaly detection with severity levels and reasoning

### Mood Predictor
- **Algorithms**: Random Forest, Gradient Boosting, Logistic Regression, SVM
- **Features**: Performance correlation with mood and energy levels
- **Output**: Mood categories (Low/Medium/High) and energy predictions

## 🎯 Usage

### Basic Usage
1. **Select a Player**: Choose from the dropdown or enter a new player name
2. **Input Performance Data**: Enter spike accuracy, blocks, digs, serves, and reaction time
3. **Get ML Predictions**: View real-time predictions and insights
4. **Submit Performance**: Save the data and get detailed analysis

### Advanced Features
- **ML-Powered Autofill**: Click the purple button to get AI-suggested values
- **Real-time Predictions**: Watch predictions update as you type
- **Player Analysis**: View historical data and trends for any player
- **Anomaly Detection**: Identify unusual performance patterns
- **Mood Insights**: Understand the relationship between performance and player mood

### API Endpoints
- `POST /api/predict` - Performance prediction
- `POST /api/anomaly` - Anomaly detection
- `POST /api/mood` - Mood prediction
- `GET /api/player/<id>` - Player data
- `GET /api/insights` - Dataset insights
- `POST /api/retrain` - Retrain models

## 📈 Visualizations

### Performance Trends Chart
- Line chart showing performance scores over time
- Spike accuracy correlation
- Trend analysis and predictions

### Skills Radar Chart
- Radar chart displaying current skill levels
- Multiple performance metrics in one view
- Easy comparison across different skills

### Anomaly Detection Chart
- Scatter plot highlighting anomalous performances
- Statistical outlier detection
- Visual representation of unusual patterns

## 🔧 Technical Details

### Data Generation
The system generates realistic volleyball data with:
- 150 players with varying skill levels
- 25+ matches per player
- Position-specific performance characteristics
- Realistic performance variations and trends

### Feature Engineering
- **Efficiency Metrics**: Spike, block, and dig efficiency ratios
- **Trend Analysis**: Rolling averages and performance trends
- **Player Characteristics**: Age, experience, position encoding
- **Performance Correlations**: Mood, energy, and match outcomes

### Model Training
- **Data Splitting**: Train/validation/test splits for each model type
- **Hyperparameter Tuning**: Grid search for optimal parameters
- **Model Selection**: Automatic selection of best performing algorithm
- **Persistence**: Trained models saved for quick loading

## 🎨 Customization

### Adding New Features
1. Update `scripts/data_generator.py` to include new data fields
2. Modify `scripts/data_preprocessing.py` for feature engineering
3. Update ML models in the respective model files
4. Add new API endpoints in `app.py`

### Styling
- Modify CSS in `index.html` for visual changes
- Update chart configurations in `js/main.js`
- Customize color schemes and layouts

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with Flask, scikit-learn, and Chart.js
- Inspired by real volleyball performance tracking needs
- Designed for coaches, players, and sports analysts

---

**VolleyVision AI** - Transforming volleyball performance tracking with the power of machine learning! 🏐🤖
