# Real-Time Threat Detection Backend - Techfest Demo

**Streamlined for instant predictions using pre-trained model pipeline**

## Quick Start

### 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Add Your Model

Place your trained model pipeline in the `models/` directory:
```
models/
└── model.joblib
```

**Model Requirements:**
- Should be a scikit-learn pipeline saved with `joblib.dump()`
- Must have `predict()` method
- Optionally `predict_proba()` for confidence scores

**Example of creating model.joblib:**
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# Create pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(n_estimators=100))
])

# Train on your data
pipeline.fit(X_train, y_train)

# Save
joblib.dump(pipeline, 'models/model.joblib')
```

### 3. Run the Server

```bash
python app.py
```

Server will start on `http://localhost:5000`

## API Endpoints

### Core Endpoint (Main Demo Feature)

**POST /api/upload-predict**
- Upload CSV file
- Get instant predictions
- Returns results with performance metrics

```bash
curl -X POST -F "file=@network_traffic.csv" http://localhost:5000/api/upload-predict
```

### Other Endpoints

- **GET /api/health** - Check API and model status
- **GET /api/model-info** - Get model information
- **GET /api/recent-predictions** - View prediction history
- **GET /api/stats** - Aggregated statistics
- **POST /api/clear-history** - Clear prediction history

## Response Format

```json
{
  "success": true,
  "results": {
    "total_samples": 1000,
    "predictions": [...],
    "confidence_scores": [...],
    "prediction_distribution": {
      "BENIGN": 850,
      "DDoS": 100,
      "PortScan": 50
    },
    "processing_time_seconds": 0.234,
    "samples_per_second": 4273,
    "performance_metrics": {
      "accuracy": 0.985,
      "precision": 0.982,
      "recall": 0.979,
      "f1_score": 0.980
    }
  }
}
```

## Features

✅ **Real-time Predictions** - Process CSV files instantly  
✅ **Performance Metrics** - Accuracy, precision, recall, F1-score  
✅ **Throughput Display** - Samples per second  
✅ **Prediction History** - Track recent predictions  
✅ **Confidence Scores** - Prediction confidence for each sample  
✅ **Distribution Analysis** - Attack type breakdown  

## Perfect for Techfest Demo!

- **Fast**: < 1 second for most datasets
- **Visual**: Rich metrics for demonstration
- **Simple**: One-click upload & predict
- **Professional**: Clean API responses
- **Reliable**: Handles edge cases gracefully

## Troubleshooting

**Model Not Found:**
- Add your `model.joblib` to the `models/` directory
- System will use a placeholder for demo if not found

**CSV Format:**
- All numeric features
- Optional ' Label' column for performance evaluation
- Handles missing values and infinities automatically

**Port Already in Use:**
- Change port in `config.py` → `API_PORT = 5001`
