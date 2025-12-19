"""
Data processing utilities for threat detection
"""
import pandas as pd
import numpy as np
import os
from config import DATASET_DIR, ATTACKS_DIR
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataProcessor:
    """Handles data loading, preprocessing, and feature engineering"""
    
    def __init__(self):
        self.data = None
        self.label_column = ' Label'  # Note: CIC-IDS2017 has space before Label
        
    def load_dataset_files(self):
        """Load all CSV files from the Dataset directory"""
        try:
            csv_files = [f for f in os.listdir(DATASET_DIR) if f.endswith('.csv')]
            
            if not csv_files:
                raise FileNotFoundError("No CSV files found in Dataset directory")
            
            logger.info(f"Found {len(csv_files)} CSV files")
            
            dataframes = []
            for csv_file in csv_files:
                file_path = os.path.join(DATASET_DIR, csv_file)
                logger.info(f"Loading {csv_file}...")
                df = pd.read_csv(file_path)
                dataframes.append(df)
            
            # Combine all dataframes
            self.data = pd.concat(dataframes, ignore_index=True)
            logger.info(f"Total records loaded: {len(self.data)}")
            
            return True, "Dataset loaded successfully"
            
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            return False, str(e)
    
    def preprocess_data(self, df=None):
        """Preprocess the data: clean, handle missing values, remove duplicates"""
        try:
            if df is None:
                df = self.data
            
            if df is None:
                return None, "No data loaded"
            
            # Make a copy
            processed_df = df.copy()
            
            # Remove rows with infinity values
            processed_df = processed_df.replace([np.inf, -np.inf], np.nan)
            
            # Drop rows with missing values
            initial_rows = len(processed_df)
            processed_df = processed_df.dropna()
            logger.info(f"Removed {initial_rows - len(processed_df)} rows with missing values")
            
            # Remove duplicate rows
            initial_rows = len(processed_df)
            processed_df = processed_df.drop_duplicates()
            logger.info(f"Removed {initial_rows - len(processed_df)} duplicate rows")
            
            # Remove constant columns (if any)
            nunique = processed_df.nunique()
            cols_to_drop = nunique[nunique == 1].index
            if len(cols_to_drop) > 0:
                processed_df = processed_df.drop(cols_to_drop, axis=1)
                logger.info(f"Removed {len(cols_to_drop)} constant columns")
            
            return processed_df, "Data preprocessed successfully"
            
        except Exception as e:
            logger.error(f"Error preprocessing data: {str(e)}")
            return None, str(e)
    
    def extract_attack_types(self, df=None):
        """Extract unique attack types from the dataset"""
        try:
            if df is None:
                df = self.data
            
            if df is None or self.label_column not in df.columns:
                return [], "No data or label column not found"
            
            attack_types = df[self.label_column].unique().tolist()
            logger.info(f"Found {len(attack_types)} attack types: {attack_types}")
            
            return attack_types, "Attack types extracted successfully"
            
        except Exception as e:
            logger.error(f"Error extracting attack types: {str(e)}")
            return [], str(e)
    
    def get_attack_statistics(self, df=None):
        """Get statistics about attacks in the dataset"""
        try:
            if df is None:
                df = self.data
            
            if df is None:
                return {}, "No data loaded"
            
            # Count by attack type
            attack_counts = df[self.label_column].value_counts().to_dict()
            
            # Calculate percentages
            total = len(df)
            attack_percentages = {k: (v / total) * 100 for k, v in attack_counts.items()}
            
            stats = {
                'total_records': total,
                'attack_counts': attack_counts,
                'attack_percentages': attack_percentages,
                'num_attack_types': len(attack_counts)
            }
            
            return stats, "Statistics generated successfully"
            
        except Exception as e:
            logger.error(f"Error getting statistics: {str(e)}")
            return {}, str(e)
    
    def filter_by_attack(self, attack_type, df=None):
        """Filter dataset by specific attack type"""
        try:
            if df is None:
                df = self.data
            
            if df is None:
                return None, "No data loaded"
            
            filtered_df = df[df[self.label_column] == attack_type].copy()
            logger.info(f"Filtered {len(filtered_df)} records for attack type: {attack_type}")
            
            return filtered_df, f"Filtered data for {attack_type}"
            
        except Exception as e:
            logger.error(f"Error filtering by attack: {str(e)}")
            return None, str(e)
    
    def save_attack_data(self, attack_type, df):
        """Save attack-specific data to file"""
        try:
            filename = os.path.join(ATTACKS_DIR, f"{attack_type.replace(' ', '_')}.csv")
            df.to_csv(filename, index=False)
            logger.info(f"Saved attack data to {filename}")
            return True, f"Saved to {filename}"
        except Exception as e:
            logger.error(f"Error saving attack data: {str(e)}")
            return False, str(e)
    
    def get_feature_columns(self, df=None):
        """Get list of feature columns (excluding label)"""
        if df is None:
            df = self.data
        
        if df is None:
            return []
        
        # Exclude label column and non-numeric columns
        feature_cols = [col for col in df.columns if col != self.label_column]
        
        # Only keep numeric columns
        numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        
        return numeric_cols
