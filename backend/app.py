"""
Streamlined Flask REST API for Real-Time Threat Detection Demo
Focuses on CSV upload → Model prediction → Results display
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import os
import logging
from werkzeug.utils import secure_filename
from datetime import datetime

from config import API_HOST, API_PORT, DEBUG, UPLOADS_DIR, ALLOWED_EXTENSIONS
from prediction_service import PredictionService
from traffic_generator import TrafficGenerator

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize prediction service
prediction_service = PredictionService()

# Initialize traffic generator
traffic_generator = TrafficGenerator(prediction_service)

# Store recent predictions for demo
recent_predictions = []


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    model_info = prediction_service.get_model_info()
    
    return jsonify({
        'status': 'healthy',
        'message': 'Threat Detection API (Real-Time Demo)',
        'version': '2.0.0',
        'model_loaded': model_info['loaded'],
        'timestamp': datetime.now().isoformat()
    }), 200


@app.route('/api/model-info', methods=['GET'])
def get_model_info():
    """Get information about the loaded model"""
    info = prediction_service.get_model_info()
    
    return jsonify({
        'success': True,
        'model_info': info
    }), 200


@app.route('/api/upload-predict', methods=['POST'])
def upload_and_predict():
    """
    Main endpoint: Upload CSV and get instant predictions
    This is the core functionality for the techfest demo
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Only CSV files allowed'}), 400
        
        # Save file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(UPLOADS_DIR, unique_filename)
        file.save(filepath)
        
        logger.info(f"📁 File uploaded: {unique_filename}")
        
        # Get instant predictions
        results, message = prediction_service.predict_from_csv(filepath)
        
        if results is None:
            return jsonify({
                'success': False,
                'error': message
            }), 400
        
        # Store in recent predictions
        prediction_record = {
            'filename': filename,
            'timestamp': results['timestamp'],
            'total_samples': results['total_samples'],
            'processing_time': results['processing_time_seconds'],
            'distribution': results['prediction_distribution']
        }
        recent_predictions.insert(0, prediction_record)
        if len(recent_predictions) > 10:  # Keep last 10
            recent_predictions.pop()
        
        logger.info(f"✅ Prediction complete: {results['total_samples']} samples in {results['processing_time_seconds']:.2f}s")
        
        return jsonify({
            'success': True,
            'message': 'Prediction completed successfully',
            'filename': filename,
            'results': results
        }), 200
        
    except Exception as e:
        logger.error(f"❌ Error in upload-predict: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/sample-predict', methods=['POST'])
def sample_predict():
    """
    Predict on a sample of data (for quick demo)
    Expects JSON with sample data
    """
    try:
        data = request.get_json()
        
        if 'samples' not in data:
            return jsonify({'error': 'No samples provided'}), 400
        
        # Convert to DataFrame
        df = pd.DataFrame(data['samples'])
        
        # Save as temporary CSV
        temp_filename = f"temp_sample_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        temp_filepath = os.path.join(UPLOADS_DIR, temp_filename)
        df.to_csv(temp_filepath, index=False)
        
        # Get predictions
        results, message = prediction_service.predict_from_csv(temp_filepath)
        
        # Clean up temp file
        os.remove(temp_filepath)
        
        if results is None:
            return jsonify({
                'success': False,
                'error': message
            }), 400
        
        return jsonify({
            'success': True,
            'message': 'Prediction completed',
            'results': results
        }), 200
        
    except Exception as e:
        logger.error(f"Error in sample predict: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/recent-predictions', methods=['GET'])
def get_recent_predictions():
    """Get recent prediction history"""
    return jsonify({
        'success': True,
        'predictions': recent_predictions,
        'count': len(recent_predictions)
    }), 200


@app.route('/api/stats', methods=['GET'])
def get_statistics():
    """Get overall statistics from recent predictions"""
    if not recent_predictions:
        return jsonify({
            'success': True,
            'message': 'No predictions yet',
            'statistics': {
                'total_predictions': 0,
                'total_samples_processed': 0,
                'average_processing_time': 0
            }
        }), 200
    
    total_samples = sum(p['total_samples'] for p in recent_predictions)
    avg_time = sum(p['processing_time'] for p in recent_predictions) / len(recent_predictions)
    
    # Aggregate distribution
    aggregated_distribution = {}
    for pred in recent_predictions:
        for attack_type, count in pred['distribution'].items():
            aggregated_distribution[attack_type] = aggregated_distribution.get(attack_type, 0) + count
    
    statistics = {
        'total_predictions': len(recent_predictions),
        'total_samples_processed': total_samples,
        'average_processing_time': avg_time,
        'threat_distribution': aggregated_distribution
    }
    
    return jsonify({
        'success': True,
        'statistics': statistics
    }), 200


@app.route('/api/clear-history', methods=['POST'])
def clear_history():
    """Clear prediction history"""
    global recent_predictions
    recent_predictions = []
    
    return jsonify({
        'success': True,
        'message': 'History cleared'
    }), 200


@app.route('/api/simulation/start', methods=['POST'])
def start_simulation():
    """Start traffic simulation for a network"""
    try:
        data = request.get_json()
        network_id = data.get('network_id')
        traffic_type = data.get('type', 'benign')
        
        if not network_id:
            return jsonify({'error': 'Network ID required'}), 400
            
        traffic_generator.start_simulation(network_id, traffic_type)
        
        return jsonify({
            'success': True,
            'message': f'Simulation started for Network {network_id} ({traffic_type})'
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/simulation/stop', methods=['POST'])
def stop_simulation():
    """Stop traffic simulation for a network"""
    try:
        data = request.get_json()
        network_id = data.get('network_id')
        
        if not network_id:
            return jsonify({'error': 'Network ID required'}), 400
            
        traffic_generator.stop_simulation(network_id)
        
        return jsonify({
            'success': True,
            'message': f'Simulation stopped for Network {network_id}'
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/simulation/status', methods=['GET'])
def get_simulation_status():
    """Get status of running simulations"""
    try:
        status = traffic_generator.get_status()
        return jsonify({
            'success': True,
            'status': status
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("🚀 CyberShield AI - Real-Time Threat Detection Demo")
    logger.info("=" * 60)
    logger.info(f"📍 Starting API on {API_HOST}:{API_PORT}")
    
    model_info = prediction_service.get_model_info()
    if model_info['loaded']:
        logger.info(f"✅ Model loaded: {model_info.get('model_type', 'Unknown')}")
    else:
        logger.warning("⚠️  No model loaded - using placeholder")
        logger.warning("   Add your trained model.joblib to the models/ directory")
    
    logger.info("=" * 60)
    
    app.run(host=API_HOST, port=API_PORT, debug=DEBUG)
