"""
Configuration settings for the Threat Detection Backend
"""
import os

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

# Dataset paths
DATASET_DIR = os.path.join(PROJECT_ROOT, 'Dataset')
ATTACKS_DIR = os.path.join(PROJECT_ROOT, 'attacks')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
UPLOADS_DIR = os.path.join(BASE_DIR, 'uploads')

# Ensure directories exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(ATTACKS_DIR, exist_ok=True)

# Model configuration
RANDOM_STATE = 42
TEST_SIZE = 0.3
CV_FOLDS = 5

# Machine Learning Algorithms
ML_ALGORITHMS = {
    'RandomForest': 'Random Forest',
    'SVM': 'Support Vector Machine',
    'KNN': 'K-Nearest Neighbors',
    'DecisionTree': 'Decision Tree',
    'NaiveBayes': 'Naive Bayes',
    'LogisticRegression': 'Logistic Regression',
    'GradientBoosting': 'Gradient Boosting'
}

# Feature selection
N_TOP_FEATURES = 4

# Attack types in CIC-IDS2017 dataset
ATTACK_TYPES = [
    'BENIGN',
    'DDoS',
    'PortScan',
    'Bot',
    'Infiltration',
    'Web Attack',
    'Brute Force',
    'DoS',
    'Heartbleed',
    'SSH-Patator',
    'FTP-Patator'
]

# API Configuration
API_HOST = '0.0.0.0'
API_PORT = 5000
DEBUG = True

# File upload settings
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
ALLOWED_EXTENSIONS = {'csv'}
