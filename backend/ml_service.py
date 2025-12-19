"""
Machine Learning service for threat detection
"""
import pandas as pd
import numpy as np
import pickle
import os
import logging
from datetime import datetime

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

from config import MODELS_DIR, RANDOM_STATE, TEST_SIZE, CV_FOLDS, N_TOP_FEATURES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import imbalanced-learn, but make it optional
try:
    from imblearn.under_sampling import RandomUnderSampler
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    logger.warning("imbalanced-learn not available. Undersampling feature will be disabled.")



class MLService:
    """Machine Learning service for training and prediction"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.feature_selectors = {}
        self.selected_features = {}
        
    def get_algorithm(self, algorithm_name):
        """Get ML algorithm instance by name"""
        algorithms = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
            'SVM': SVC(kernel='rbf', random_state=RANDOM_STATE),
            'KNN': KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
            'DecisionTree': DecisionTreeClassifier(random_state=RANDOM_STATE),
            'NaiveBayes': GaussianNB(),
            'LogisticRegression': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000, n_jobs=-1),
            'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE)
        }
        return algorithms.get(algorithm_name)
    
    def feature_selection(self, X, y, n_features=N_TOP_FEATURES):
        """Select top N features using ANOVA F-test"""
        try:
            selector = SelectKBest(score_func=f_classif, k=min(n_features, X.shape[1]))
            X_selected = selector.fit_transform(X, y)
            
            # Get selected feature names
            selected_indices = selector.get_support(indices=True)
            selected_features = X.columns[selected_indices].tolist()
            
            logger.info(f"Selected features: {selected_features}")
            
            return X_selected, selected_features, selector
            
        except Exception as e:
            logger.error(f"Error in feature selection: {str(e)}")
            return X, X.columns.tolist(), None
    
    def undersample_data(self, X, y):
        """Handle imbalanced data using undersampling"""
        if not IMBLEARN_AVAILABLE:
            logger.warning("Undersampling requested but imbalanced-learn is not available")
            return X, y
            
        try:
            rus = RandomUnderSampler(random_state=RANDOM_STATE)
            X_resampled, y_resampled = rus.fit_resample(X, y)
            
            logger.info(f"Undersampled from {len(y)} to {len(y_resampled)} samples")
            
            return X_resampled, y_resampled
            
        except Exception as e:
            logger.error(f"Error in undersampling: {str(e)}")
            return X, y
    
    def train_model(self, df, algorithm='RandomForest', use_feature_selection=True, 
                   use_undersampling=False, label_column=' Label'):
        """Train a model on the provided dataset"""
        try:
            logger.info(f"Training {algorithm} model...")
            
            # Separate features and labels
            if label_column not in df.columns:
                return None, "Label column not found in dataset"
            
            X = df.drop(columns=[label_column])
            y = df[label_column]
            
            # Keep only numeric features
            X = X.select_dtypes(include=[np.number])
            
            if X.empty:
                return None, "No numeric features found"
            
            # Encode labels
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            
            # Feature selection
            selected_features = X.columns.tolist()
            selector = None
            
            if use_feature_selection and X.shape[1] > N_TOP_FEATURES:
                X_selected, selected_features, selector = self.feature_selection(X, y_encoded, N_TOP_FEATURES)
                X = pd.DataFrame(X_selected, columns=selected_features)
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Undersampling
            if use_undersampling:
                X_scaled, y_encoded = self.undersample_data(X_scaled, y_encoded)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_encoded, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_encoded
            )
            
            # Get algorithm
            model = self.get_algorithm(algorithm)
            if model is None:
                return None, f"Unknown algorithm: {algorithm}"
            
            # Train model
            logger.info(f"Training on {len(X_train)} samples...")
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            # Cross-validation score
            try:
                cv_scores = cross_val_score(model, X_train, y_train, cv=min(CV_FOLDS, len(X_train)), n_jobs=-1)
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()
            except:
                cv_mean = 0
                cv_std = 0
            
            # Save model components
            model_id = f"{algorithm}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            self.models[model_id] = model
            self.scalers[model_id] = scaler
            self.label_encoders[model_id] = le
            self.feature_selectors[model_id] = selector
            self.selected_features[model_id] = selected_features
            
            # Save to disk
            self.save_model(model_id)
            
            results = {
                'model_id': model_id,
                'algorithm': algorithm,
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'cv_mean': float(cv_mean),
                'cv_std': float(cv_std),
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'selected_features': selected_features,
                'num_classes': len(le.classes_),
                'classes': le.classes_.tolist()
            }
            
            logger.info(f"Model trained successfully. Accuracy: {accuracy:.4f}")
            
            return results, "Model trained successfully"
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            return None, str(e)
    
    def predict(self, model_id, data):
        """Make predictions using a trained model"""
        try:
            if model_id not in self.models:
                # Try to load from disk
                success, msg = self.load_model(model_id)
                if not success:
                    return None, f"Model {model_id} not found"
            
            model = self.models[model_id]
            scaler = self.scalers[model_id]
            le = self.label_encoders[model_id]
            selected_features = self.selected_features[model_id]
            
            # Prepare data
            if isinstance(data, pd.DataFrame):
                X = data[selected_features]
            else:
                X = pd.DataFrame([data])[selected_features]
            
            # Scale
            X_scaled = scaler.transform(X)
            
            # Predict
            predictions = model.predict(X_scaled)
            probabilities = model.predict_proba(X_scaled) if hasattr(model, 'predict_proba') else None
            
            # Decode labels
            predicted_labels = le.inverse_transform(predictions)
            
            results = {
                'predictions': predicted_labels.tolist(),
                'probabilities': probabilities.tolist() if probabilities is not None else None,
                'classes': le.classes_.tolist()
            }
            
            return results, "Prediction successful"
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            return None, str(e)
    
    def save_model(self, model_id):
        """Save model to disk"""
        try:
            model_path = os.path.join(MODELS_DIR, f"{model_id}.pkl")
            
            model_data = {
                'model': self.models[model_id],
                'scaler': self.scalers[model_id],
                'label_encoder': self.label_encoders[model_id],
                'feature_selector': self.feature_selectors.get(model_id),
                'selected_features': self.selected_features[model_id]
            }
            
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"Model saved to {model_path}")
            return True, f"Model saved to {model_path}"
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return False, str(e)
    
    def load_model(self, model_id):
        """Load model from disk"""
        try:
            model_path = os.path.join(MODELS_DIR, f"{model_id}.pkl")
            
            if not os.path.exists(model_path):
                return False, f"Model file not found: {model_path}"
            
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.models[model_id] = model_data['model']
            self.scalers[model_id] = model_data['scaler']
            self.label_encoders[model_id] = model_data['label_encoder']
            self.feature_selectors[model_id] = model_data.get('feature_selector')
            self.selected_features[model_id] = model_data['selected_features']
            
            logger.info(f"Model loaded from {model_path}")
            return True, f"Model loaded successfully"
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False, str(e)
    
    def list_models(self):
        """List all available models"""
        try:
            model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith('.pkl')]
            models_info = []
            
            for model_file in model_files:
                model_id = model_file.replace('.pkl', '')
                model_path = os.path.join(MODELS_DIR, model_file)
                
                # Get file stats
                stats = os.stat(model_path)
                
                models_info.append({
                    'model_id': model_id,
                    'file_path': model_path,
                    'size_bytes': stats.st_size,
                    'created': datetime.fromtimestamp(stats.st_ctime).isoformat()
                })
            
            return models_info, "Models listed successfully"
            
        except Exception as e:
            logger.error(f"Error listing models: {str(e)}")
            return [], str(e)
