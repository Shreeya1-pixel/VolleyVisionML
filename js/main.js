// VolleyVision ML - Enhanced with Dynamic Tabs and Advanced Visualizations
class VolleyVisionML {
    constructor() {
        this.apiBase = 'http://localhost:5001/api';
        this.charts = {};
        this.currentTab = 'dashboard';
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.loadInitialData();
        this.setupTabNavigation();
    }

    setupEventListeners() {
        // Form submission
        const form = document.getElementById('statsForm');
        if (form) {
            form.addEventListener('submit', (e) => {
                e.preventDefault();
                this.submitPerformance();
            });
        }

        // Real-time updates
        const inputs = document.querySelectorAll('input[type="number"], input[type="text"]');
        inputs.forEach(input => {
            input.addEventListener('input', () => {
                this.updatePredictions();
            });
        });
    }

    setupTabNavigation() {
        const navBtns = document.querySelectorAll('.nav-btn');
        const tabContents = document.querySelectorAll('.tab-content');

        navBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                const targetTab = btn.getAttribute('data-tab');
                this.switchTab(targetTab);
            });
        });
    }

    switchTab(targetTab) {
        // Remove active class from all buttons and tabs
        document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(tab => tab.classList.remove('active'));
        
        // Add active class to clicked button and target tab
        document.querySelector(`[data-tab="${targetTab}"]`).classList.add('active');
        document.getElementById(targetTab).classList.add('active');
        
        this.currentTab = targetTab;
        
        // Load tab-specific content
        this.loadTabContent(targetTab);
        
        // Add smooth animation
        this.animateTabTransition();
    }

    animateTabTransition() {
        const activeTab = document.querySelector('.tab-content.active');
        activeTab.style.opacity = '0';
        activeTab.style.transform = 'translateY(20px)';
        
        setTimeout(() => {
            activeTab.style.transition = 'all 0.6s cubic-bezier(0.4, 0, 0.2, 1)';
            activeTab.style.opacity = '1';
            activeTab.style.transform = 'translateY(0)';
        }, 100);
    }

    async loadTabContent(tabName) {
        switch(tabName) {
            case 'dashboard':
                await this.loadDashboardContent();
                break;
            case 'analytics':
                await this.loadAnalyticsContent();
                break;
            case 'predictions':
                await this.loadPredictionsContent();
                break;
            case 'similarity':
                await this.loadSimilarityContent();
                break;
        }
    }

    async loadDashboardContent() {
        // Dashboard is already loaded
        this.updatePredictions();
        // Initialize dashboard charts immediately
        await this.initializeDashboardCharts();
    }

    async initializeDashboardCharts() {
        const formData = this.getFormData();
        if (formData.spikeAccuracy && formData.blocks) {
            await this.createPerformanceChart(formData);
            await this.createRadarChart(formData);
        } else {
            // Create charts with default data if form is empty
            const defaultData = {
                spikeAccuracy: 87,
                blocks: 6,
                digs: 22,
                serves: '4/12',
                reactionTime: 310
            };
            await this.createPerformanceChart(defaultData);
            await this.createRadarChart(defaultData);
        }
        
        // Initialize efficiency score with default value
        this.initializeEfficiencyScore();
    }

    initializeEfficiencyScore() {
        const efficiencyScore = document.getElementById('efficiencyScore');
        const nextMatchPrediction = document.getElementById('nextMatchPrediction');
        const improvementAreas = document.getElementById('improvementAreas');
        const confidenceLevel = document.getElementById('confidenceLevel');
        
        if (efficiencyScore) {
            efficiencyScore.textContent = '85.0';
            efficiencyScore.style.background = this.getScoreGradient(85);
        }
        
        if (nextMatchPrediction) {
            nextMatchPrediction.textContent = '88.5%';
        }
        
        if (improvementAreas) {
            improvementAreas.textContent = 'Spike Accuracy, Blocking';
        }
        
        if (confidenceLevel) {
            confidenceLevel.textContent = '85.0%';
        }
    }

    async loadAnalyticsContent() {
        await this.createTimeSeriesChart();
        await this.createAnomalyChart();
        this.updateAnomalyStatus();
    }

    async loadPredictionsContent() {
        await this.createForecastChart();
        await this.createRecommendationsChart();
        this.updateRecommendations();
    }

    async loadSimilarityContent() {
        await this.createSimilarityChart();
        this.updateSimilarPlayers();
    }

    async loadInitialData() {
        try {
            // Load insights
            const insightsResponse = await fetch(`${this.apiBase}/insights`);
            const insights = await insightsResponse.json();
            this.displayInsights(insights);

            // Load player data
            const playerResponse = await fetch(`${this.apiBase}/player/1`);
            const playerData = await playerResponse.json();
            this.displayPlayerData(playerData);

            // Initialize charts immediately
            await this.initializeDashboardCharts();

        } catch (error) {
            console.error('Error loading initial data:', error);
            // Still initialize charts even if API fails
            await this.initializeDashboardCharts();
        }
    }

    getFormData() {
        return {
            playerName: document.getElementById('playerName').value,
            spikeAccuracy: parseFloat(document.getElementById('spikeAccuracy').value),
            blocks: parseFloat(document.getElementById('blocks').value),
            digs: parseFloat(document.getElementById('digs').value),
            serves: document.getElementById('serves').value,
            reactionTime: parseFloat(document.getElementById('reactionTime').value)
        };
    }

    async submitPerformance() {
        const formData = this.getFormData();
        console.log('Submitting performance data:', formData);
        
        try {
            // Get ML predictions
            const predictionResponse = await fetch(`${this.apiBase}/predict`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(formData)
            });
            const predictions = await predictionResponse.json();
            console.log('Prediction response:', predictions);

            // Get anomaly detection
            const anomalyResponse = await fetch(`${this.apiBase}/anomaly`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(formData)
            });
            const anomaly = await anomalyResponse.json();
            console.log('Anomaly response:', anomaly);

            // Get mood prediction
            const moodResponse = await fetch(`${this.apiBase}/mood`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(formData)
            });
            const mood = await moodResponse.json();
            console.log('Mood response:', mood);

            this.displayPredictionResults(predictions, anomaly, mood);
            this.updateCharts(formData, predictions);

        } catch (error) {
            console.error('Error submitting performance:', error);
            this.showNotification('Error submitting data', 'error');
        }
    }

    async updatePredictions() {
        const formData = this.getFormData();
        
        try {
            const response = await fetch(`${this.apiBase}/predict`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(formData)
            });
            const predictions = await response.json();
            
            this.updateRealTimeDisplay(predictions);
        } catch (error) {
            console.error('Error updating predictions:', error);
        }
    }

    displayPredictionResults(predictions, anomaly, mood) {
        console.log('Displaying predictions:', predictions);
        
        // Update efficiency score
        const efficiencyScore = document.getElementById('efficiencyScore');
        if (efficiencyScore && predictions.predicted_performance) {
            efficiencyScore.textContent = `${predictions.predicted_performance.toFixed(1)}`;
            efficiencyScore.style.background = this.getScoreGradient(predictions.predicted_performance);
        }

        // Update prediction cards
        const nextMatchPrediction = document.getElementById('nextMatchPrediction');
        const improvementAreas = document.getElementById('improvementAreas');
        const confidenceLevel = document.getElementById('confidenceLevel');

        if (nextMatchPrediction && predictions.next_match_prediction) {
            const nextMatch = predictions.next_match_prediction;
            const avgPrediction = (nextMatch.spike_accuracy + nextMatch.blocks * 10 + nextMatch.digs * 2) / 3;
            nextMatchPrediction.textContent = `${avgPrediction.toFixed(1)}%`;
        }
        
        if (improvementAreas) {
            // Determine improvement areas based on current vs predicted
            const formData = this.getFormData();
            const improvements = [];
            
            if (predictions.next_match_prediction) {
                const next = predictions.next_match_prediction;
                if (next.spike_accuracy > formData.spikeAccuracy) improvements.push('Spike Accuracy');
                if (next.blocks > formData.blocks) improvements.push('Blocking');
                if (next.digs > formData.digs) improvements.push('Digging');
            }
            
            improvementAreas.textContent = improvements.length > 0 ? improvements.join(', ') : 'Maintain Current Level';
        }
        
        if (confidenceLevel && predictions.confidence) {
            confidenceLevel.textContent = `${(predictions.confidence * 100).toFixed(1)}%`;
        }

        // Show notification
        this.showNotification('ML Analytics Generated Successfully!', 'success');
    }

    updateRealTimeDisplay(predictions) {
        const efficiencyScore = document.getElementById('efficiencyScore');
        if (efficiencyScore && predictions.predicted_performance) {
            efficiencyScore.textContent = `${predictions.predicted_performance.toFixed(1)}`;
            efficiencyScore.style.background = this.getScoreGradient(predictions.predicted_performance);
        }
    }

    getScoreGradient(score) {
        if (score >= 80) return 'linear-gradient(45deg, #27ae60, #2ecc71)';
        if (score >= 60) return 'linear-gradient(45deg, #f39c12, #f1c40f)';
        return 'linear-gradient(45deg, #e74c3c, #c0392b)';
    }

    async updateCharts(formData, predictions) {
        await this.createPerformanceChart(formData, predictions);
        await this.createRadarChart(formData);
        await this.createAnomalyChart(formData);
    }

    async createPerformanceChart(formData, predictions) {
        const ctx = document.getElementById('performanceChart');
        if (!ctx) {
            console.log('Performance chart canvas not found');
            return;
        }

        if (this.charts.performance) {
            this.charts.performance.destroy();
        }

        // Use default data if formData is not provided
        const data = formData || {
            spikeAccuracy: 87,
            blocks: 6,
            digs: 22,
            serves: '4/12',
            reactionTime: 310
        };

        const serves = data.serves.split('/');
        const serveRatio = parseFloat(serves[0]) / parseFloat(serves[1]);

        // Original prediction logic
        const nextSpike = (data.spikeAccuracy + 0.2 * data.digs + 0.3 * serveRatio * 100 - 0.1 * data.reactionTime / 10).toFixed(2);
        const nextBlocks = (data.blocks + 0.5 * serveRatio * 10 + 0.2 * data.spikeAccuracy / 10).toFixed(1);

        this.charts.performance = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['Spike Accuracy (%)', 'Blocks', 'Digs', 'Serve Efficiency'],
                datasets: [
                    {
                        label: 'Current Performance',
                        data: [data.spikeAccuracy, data.blocks, data.digs, serveRatio * 100],
                        backgroundColor: 'rgba(52, 152, 219, 0.8)',
                        borderColor: 'rgba(52, 152, 219, 1)',
                        borderWidth: 2
                    },
                    {
                        label: 'ML Predicted',
                        data: [parseFloat(nextSpike), parseFloat(nextBlocks), data.digs * 1.1, serveRatio * 110],
                        backgroundColor: 'rgba(241, 196, 15, 0.8)',
                        borderColor: 'rgba(241, 196, 15, 1)',
                        borderWidth: 2
                    }
                ]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        labels: { 
                            color: '#fff', 
                            font: { size: 14, family: 'Inter' },
                            usePointStyle: true
                        }
                    },
                    title: {
                        display: true,
                        text: 'Performance Analysis',
                        color: '#fff',
                        font: { size: 16, family: 'Playfair Display' }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: { color: '#fff' },
                        grid: { color: 'rgba(255, 255, 255, 0.1)' }
                    },
                    x: {
                        ticks: { color: '#fff' },
                        grid: { color: 'rgba(255, 255, 255, 0.1)' }
                    }
                },
                animation: {
                    duration: 2000,
                    easing: 'easeInOutQuart'
                }
            }
        });
    }

    async createRadarChart(formData) {
        const ctx = document.getElementById('radarChart');
        if (!ctx) {
            console.log('Radar chart canvas not found');
            return;
        }

        if (this.charts.radar) {
            this.charts.radar.destroy();
        }

        // Use default data if formData is not provided
        const data = formData || {
            spikeAccuracy: 87,
            blocks: 6,
            digs: 22,
            serves: '4/12',
            reactionTime: 310
        };

        const serves = data.serves.split('/');
        const serveRatio = parseFloat(serves[0]) / parseFloat(serves[1]);

        this.charts.radar = new Chart(ctx, {
            type: 'radar',
            data: {
                labels: ['Spike Power', 'Blocking', 'Digging', 'Serving', 'Reaction Time', 'Overall'],
                datasets: [
                    {
                        label: 'Current Skills',
                        data: [
                            data.spikeAccuracy,
                            data.blocks * 10,
                            data.digs * 2,
                            serveRatio * 100,
                            600 - data.reactionTime,
                            (data.spikeAccuracy + data.blocks * 10 + data.digs * 2 + serveRatio * 100 + (600 - data.reactionTime)) / 5
                        ],
                        backgroundColor: 'rgba(78, 205, 196, 0.2)',
                        borderColor: 'rgba(78, 205, 196, 1)',
                        borderWidth: 3,
                        pointBackgroundColor: 'rgba(78, 205, 196, 1)',
                        pointBorderColor: '#fff',
                        pointHoverBackgroundColor: '#fff',
                        pointHoverBorderColor: 'rgba(78, 205, 196, 1)'
                    }
                ]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        labels: { 
                            color: '#fff', 
                            font: { size: 14, family: 'Inter' }
                        }
                    },
                    title: {
                        display: true,
                        text: 'Skill Radar Analysis',
                        color: '#fff',
                        font: { size: 16, family: 'Playfair Display' }
                    }
                },
                scales: {
                    r: {
                        beginAtZero: true,
                        max: 100,
                        ticks: { 
                            color: '#fff',
                            backdropColor: 'transparent'
                        },
                        grid: { color: 'rgba(255, 255, 255, 0.1)' },
                        pointLabels: { 
                            color: '#fff',
                            font: { size: 12, family: 'Inter' }
                        }
                    }
                },
                animation: {
                    duration: 2000,
                    easing: 'easeInOutQuart'
                }
            }
        });
    }

    async createTimeSeriesChart() {
        const ctx = document.getElementById('forecastChart');
        if (!ctx) return;

        if (this.charts.timeSeries) {
            this.charts.timeSeries.destroy();
        }

        // Generate time series data
        const dates = [];
        const performance = [];
        const baseDate = new Date();
        
        for (let i = 0; i < 30; i++) {
            const date = new Date(baseDate);
            date.setDate(date.getDate() - (29 - i));
            dates.push(date.toLocaleDateString());
            performance.push(65 + Math.random() * 20 + Math.sin(i * 0.3) * 5);
        }

        this.charts.timeSeries = new Chart(ctx, {
            type: 'line',
            data: {
                labels: dates,
                datasets: [
                    {
                        label: 'Performance Trend',
                        data: performance,
                        borderColor: 'rgba(255, 107, 107, 1)',
                        backgroundColor: 'rgba(255, 107, 107, 0.1)',
                        borderWidth: 3,
                        fill: true,
                        tension: 0.4
                    },
                    {
                        label: 'Forecast (Next 7 days)',
                        data: [...Array(23).fill(null), ...performance.slice(-7).map(p => p + Math.random() * 5 - 2.5)],
                        borderColor: 'rgba(78, 205, 196, 1)',
                        backgroundColor: 'rgba(78, 205, 196, 0.1)',
                        borderWidth: 3,
                        borderDash: [5, 5],
                        fill: false,
                        tension: 0.4
                    }
                ]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        labels: { 
                            color: '#fff', 
                            font: { size: 14, family: 'Inter' }
                        }
                    },
                    title: {
                        display: true,
                        text: 'Performance Forecasting',
                        color: '#fff',
                        font: { size: 16, family: 'Playfair Display' }
                    }
                },
                scales: {
                    y: {
                        ticks: { color: '#fff' },
                        grid: { color: 'rgba(255, 255, 255, 0.1)' }
                    },
                    x: {
                        ticks: { color: '#fff' },
                        grid: { color: 'rgba(255, 255, 255, 0.1)' }
                    }
                },
                animation: {
                    duration: 2000,
                    easing: 'easeInOutQuart'
                }
            }
        });
    }

    async createAnomalyChart() {
        const ctx = document.getElementById('anomalyChart');
        if (!ctx) return;

        if (this.charts.anomaly) {
            this.charts.anomaly.destroy();
        }

        // Generate anomaly data
        const data = Array.from({length: 50}, () => 70 + Math.random() * 20);
        const anomalies = data.map((value, index) => 
            Math.random() > 0.9 ? value + (Math.random() > 0.5 ? 30 : -30) : value
        );

        this.charts.anomaly = new Chart(ctx, {
            type: 'scatter',
            data: {
                datasets: [
                    {
                        label: 'Normal Performance',
                        data: data.map((value, index) => ({x: index, y: value})),
                        backgroundColor: 'rgba(78, 205, 196, 0.6)',
                        borderColor: 'rgba(78, 205, 196, 1)',
                        borderWidth: 1
                    },
                    {
                        label: 'Anomalies Detected',
                        data: anomalies.filter((value, index) => Math.abs(value - data[index]) > 15)
                            .map((value, index) => ({x: index, y: value})),
                        backgroundColor: 'rgba(255, 107, 107, 0.8)',
                        borderColor: 'rgba(255, 107, 107, 1)',
                        borderWidth: 2
                    }
                ]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        labels: { 
                            color: '#fff', 
                            font: { size: 14, family: 'Inter' }
                        }
                    },
                    title: {
                        display: true,
                        text: 'Anomaly Detection',
                        color: '#fff',
                        font: { size: 16, family: 'Playfair Display' }
                    }
                },
                scales: {
                    y: {
                        ticks: { color: '#fff' },
                        grid: { color: 'rgba(255, 255, 255, 0.1)' }
                    },
                    x: {
                        ticks: { color: '#fff' },
                        grid: { color: 'rgba(255, 255, 255, 0.1)' }
                    }
                },
                animation: {
                    duration: 2000,
                    easing: 'easeInOutQuart'
                }
            }
        });
    }

    async createForecastChart() {
        const ctx = document.getElementById('forecastChart2');
        if (!ctx) return;

        if (this.charts.forecast) {
            this.charts.forecast.destroy();
        }

        // Generate forecast data
        const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'];
        const current = [75, 78, 82, 79, 85, 88];
        const predicted = [88, 91, 89, 93, 90, 95];

        this.charts.forecast = new Chart(ctx, {
            type: 'line',
            data: {
                labels: months,
                datasets: [
                    {
                        label: 'Current Performance',
                        data: current,
                        borderColor: 'rgba(52, 152, 219, 1)',
                        backgroundColor: 'rgba(52, 152, 219, 0.1)',
                        borderWidth: 3,
                        fill: false
                    },
                    {
                        label: 'AI Predicted',
                        data: predicted,
                        borderColor: 'rgba(241, 196, 15, 1)',
                        backgroundColor: 'rgba(241, 196, 15, 0.1)',
                        borderWidth: 3,
                        borderDash: [5, 5],
                        fill: false
                    }
                ]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        labels: { 
                            color: '#fff', 
                            font: { size: 14, family: 'Inter' }
                        }
                    },
                    title: {
                        display: true,
                        text: '6-Month Performance Forecast',
                        color: '#fff',
                        font: { size: 16, family: 'Playfair Display' }
                    }
                },
                scales: {
                    y: {
                        ticks: { color: '#fff' },
                        grid: { color: 'rgba(255, 255, 255, 0.1)' }
                    },
                    x: {
                        ticks: { color: '#fff' },
                        grid: { color: 'rgba(255, 255, 255, 0.1)' }
                    }
                },
                animation: {
                    duration: 2000,
                    easing: 'easeInOutQuart'
                }
            }
        });
    }

    async createRecommendationsChart() {
        const ctx = document.getElementById('recommendationsChart');
        if (!ctx) return;

        if (this.charts.recommendations) {
            this.charts.recommendations.destroy();
        }

        this.charts.recommendations = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Spike Training', 'Blocking Drills', 'Serve Practice', 'Reaction Time', 'Team Coordination'],
                datasets: [{
                    data: [30, 25, 20, 15, 10],
                    backgroundColor: [
                        'rgba(255, 107, 107, 0.8)',
                        'rgba(78, 205, 196, 0.8)',
                        'rgba(241, 196, 15, 0.8)',
                        'rgba(52, 152, 219, 0.8)',
                        'rgba(155, 89, 182, 0.8)'
                    ],
                    borderColor: [
                        'rgba(255, 107, 107, 1)',
                        'rgba(78, 205, 196, 1)',
                        'rgba(241, 196, 15, 1)',
                        'rgba(52, 152, 219, 1)',
                        'rgba(155, 89, 182, 1)'
                    ],
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: { 
                            color: '#fff', 
                            font: { size: 12, family: 'Inter' },
                            usePointStyle: true
                        }
                    },
                    title: {
                        display: true,
                        text: 'Training Focus Distribution',
                        color: '#fff',
                        font: { size: 16, family: 'Playfair Display' }
                    }
                },
                animation: {
                    duration: 2000,
                    easing: 'easeInOutQuart'
                }
            }
        });
    }

    async createSimilarityChart() {
        const ctx = document.getElementById('similarPlayersChart');
        if (!ctx) return;

        if (this.charts.similarity) {
            this.charts.similarity.destroy();
        }

        this.charts.similarity = new Chart(ctx, {
            type: 'radar',
            data: {
                labels: ['Spike Power', 'Blocking', 'Digging', 'Serving', 'Reaction Time', 'Team Play'],
                datasets: [
                    {
                        label: 'Your Profile',
                        data: [85, 70, 80, 75, 85, 90],
                        backgroundColor: 'rgba(78, 205, 196, 0.2)',
                        borderColor: 'rgba(78, 205, 196, 1)',
                        borderWidth: 3
                    },
                    {
                        label: 'Most Similar Player',
                        data: [82, 72, 78, 77, 83, 88],
                        backgroundColor: 'rgba(255, 107, 107, 0.2)',
                        borderColor: 'rgba(255, 107, 107, 1)',
                        borderWidth: 3
                    }
                ]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        labels: { 
                            color: '#fff', 
                            font: { size: 14, family: 'Inter' }
                        }
                    },
                    title: {
                        display: true,
                        text: 'Player Similarity Analysis',
                        color: '#fff',
                        font: { size: 16, family: 'Playfair Display' }
                    }
                },
                scales: {
                    r: {
                        beginAtZero: true,
                        max: 100,
                        ticks: { color: '#fff' },
                        grid: { color: 'rgba(255, 255, 255, 0.1)' },
                        pointLabels: { color: '#fff' }
                    }
                },
                animation: {
                    duration: 2000,
                    easing: 'easeInOutQuart'
                }
            }
        });
    }

    updateAnomalyStatus() {
        const anomalyStatus = document.getElementById('anomalyStatus');
        const anomalyExplanation = document.getElementById('anomalyExplanation');
        
        if (anomalyStatus) {
            anomalyStatus.textContent = 'Anomaly detected in recent performance';
            anomalyStatus.style.color = '#ff6b6b';
        }
        
        if (anomalyExplanation) {
            anomalyExplanation.innerHTML = `
                <p><strong>Spike Accuracy:</strong> 15% above team average</p>
                <p><strong>Blocking Performance:</strong> Exceptional timing detected</p>
                <p><strong>Recommendation:</strong> Maintain current training intensity</p>
            `;
        }
    }

    updateRecommendations() {
        const recommendationsList = document.getElementById('recommendationsList');
        if (recommendationsList) {
            recommendationsList.innerHTML = `
                <li>Focus on spike accuracy training (30% priority)</li>
                <li>Increase blocking drills intensity (25% priority)</li>
                <li>Practice serve consistency (20% priority)</li>
                <li>Improve reaction time with agility drills (15% priority)</li>
                <li>Enhance team coordination skills (10% priority)</li>
            `;
        }
    }

    updateSimilarPlayers() {
        const similarPlayer1 = document.getElementById('similarPlayer1');
        const similarPlayer2 = document.getElementById('similarPlayer2');
        
        if (similarPlayer1) {
            similarPlayer1.querySelector('.player-name').textContent = 'Alex Johnson';
            similarPlayer1.querySelector('.similarity-score').textContent = '94.2% similarity';
        }
        
        if (similarPlayer2) {
            similarPlayer2.querySelector('.player-name').textContent = 'Maria Rodriguez';
            similarPlayer2.querySelector('.similarity-score').textContent = '87.6% similarity';
        }
    }

    displayInsights(insights) {
        // Insights are displayed in the dashboard
    }

    displayPlayerData(playerData) {
        // Player data is displayed in the dashboard
    }

    showNotification(message, type = 'info') {
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.textContent = message;
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 15px 20px;
            border-radius: 10px;
            color: white;
            font-weight: 600;
            z-index: 1000;
            animation: slideIn 0.3s ease-out;
            background: ${type === 'success' ? 'linear-gradient(45deg, #27ae60, #2ecc71)' : 
                        type === 'error' ? 'linear-gradient(45deg, #e74c3c, #c0392b)' : 
                        'linear-gradient(45deg, #3498db, #2980b9)'};
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3);
        `;

        document.body.appendChild(notification);

        setTimeout(() => {
            notification.style.animation = 'slideOut 0.3s ease-in';
            setTimeout(() => {
                document.body.removeChild(notification);
            }, 300);
        }, 3000);
    }
}

// Original functions - keeping the exact same functionality
function autoFill() {
    document.getElementById('playerName').value = 'Shreeya Gupta';
    document.getElementById('spikeAccuracy').value = 87;
    document.getElementById('blocks').value = 6;
    document.getElementById('digs').value = 22;
    document.getElementById('serves').value = '4/12';
    document.getElementById('reactionTime').value = 310;
}

function submitForm() {
    const name = document.getElementById('playerName').value;
    const spike = parseFloat(document.getElementById('spikeAccuracy').value);
    const blocks = parseFloat(document.getElementById('blocks').value);
    const digs = parseFloat(document.getElementById('digs').value);
    const serves = document.getElementById('serves').value.split('/');
    const reaction = parseFloat(document.getElementById('reactionTime').value);

    const serveRatio = parseFloat(serves[0]) / parseFloat(serves[1]);

    // Original ML-like prediction (keeping the exact same logic)
    const nextSpike = (spike + 0.2 * digs + 0.3 * serveRatio * 100 - 0.1 * reaction / 10).toFixed(2);
    const nextBlocks = (blocks + 0.5 * serveRatio * 10 + 0.2 * spike / 10).toFixed(1);

    showGraph(spike, blocks, nextSpike, nextBlocks);
    alert(`Future performance predicted. Check graph below.`);
}

// Original chart functionality - keeping exactly the same
let chart;

function showGraph(currentSpike, currentBlocks, predictedSpike, predictedBlocks) {
    const ctx = document.getElementById('performanceChart').getContext('2d');

    if (chart) chart.destroy(); // Reset old chart

    chart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Spike Accuracy (%)', 'Blocks'],
            datasets: [
                {
                    label: 'Current',
                    data: [currentSpike, currentBlocks],
                    backgroundColor: 'rgba(52, 152, 219, 0.6)'
                },
                {
                    label: 'Predicted',
                    data: [predictedSpike, predictedBlocks],
                    backgroundColor: 'rgba(241, 196, 15, 0.8)'
                }
            ]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    labels: { color: '#fff', font: { size: 14 } }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: { color: '#fff' }
                },
                x: {
                    ticks: { color: '#fff' }
                }
            }
        }
    });
}

// Initialize VolleyVision ML when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    window.volleyVision = new VolleyVisionML();
    
    // Add CSS animations
    const style = document.createElement('style');
    style.textContent = `
        @keyframes slideIn {
            from {
                transform: translateX(100%);
                opacity: 0;
            }
            to {
                transform: translateX(0);
                opacity: 1;
            }
        }
        
        @keyframes slideOut {
            from {
                transform: translateX(0);
                opacity: 1;
            }
            to {
                transform: translateX(100%);
                opacity: 0;
            }
        }
        
        .tab-content {
            transition: all 0.6s cubic-bezier(0.4, 0, 0.2, 1);
        }
        
        .chart-container {
            transition: all 0.3s ease;
        }
        
        .chart-container:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
        }
    `;
    document.head.appendChild(style);
});