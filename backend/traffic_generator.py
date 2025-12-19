"""
Synthetic Traffic Generator for Cyber Security Demo
Generates realistic network traffic data based on templates from the actual dataset.
"""
import pandas as pd
import numpy as np
import threading
import time
import logging
import os
import random
from config import DATASET_DIR

logger = logging.getLogger(__name__)

class TrafficGenerator:
    def __init__(self, prediction_service):
        self.prediction_service = prediction_service
        self.active_simulations = {}
        self.templates = {
            'benign': None,
            'attack': None
        }
        self.running = False
        
        # Load templates
        self._load_templates()

    def _load_templates(self):
        """Load template data from dataset files to use as base for generation"""
        try:
            # Load Benign template (Monday)
            monday_path = os.path.join(DATASET_DIR, 'Monday-WorkingHours.pcap_ISCX.csv')
            if os.path.exists(monday_path):
                df = pd.read_csv(monday_path, nrows=1000)
                # Clean column names (strip spaces)
                df.columns = df.columns.str.strip()
                if 'Label' in df.columns:
                    self.templates['benign'] = df[df['Label'] == 'BENIGN'].drop(columns=['Label'])
                logger.info(f"Loaded {len(self.templates['benign'])} benign templates")

            # Load Attack template (DDoS - Friday)
            friday_path = os.path.join(DATASET_DIR, 'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv')
            if os.path.exists(friday_path):
                # Read enough rows to reach the attack data (starts ~18k)
                df = pd.read_csv(friday_path, nrows=25000)
                df.columns = df.columns.str.strip()
                if 'Label' in df.columns:
                    self.templates['attack'] = df[df['Label'] == 'DDoS'].drop(columns=['Label'])
                logger.info(f"Loaded {len(self.templates['attack'])} attack templates")
                
        except Exception as e:
            logger.error(f"Error loading templates: {e}")
            # Fallback will be handled in generation

    def generate_traffic(self, traffic_type='benign', batch_size=1):
        """
        Generate synthetic traffic based on templates with jitter
        Returns: DataFrame with generated samples
        """
        template_key = 'attack' if traffic_type == 'attack' else 'benign'
        template_df = self.templates[template_key]
        
        if template_df is None or template_df.empty:
            logger.warning(f"No template for {traffic_type}, returning zeros")
            return pd.DataFrame(np.zeros((batch_size, 78))) # Approximate shape

        # Sample from templates
        samples = template_df.sample(n=batch_size, replace=True).copy()
        
        # Add jitter to numeric columns to make it "new" data
        numeric_cols = samples.select_dtypes(include=[np.number]).columns
        
        # 1-5% random variation
        jitter = np.random.uniform(0.95, 1.05, size=samples[numeric_cols].shape)
        samples[numeric_cols] = samples[numeric_cols] * jitter
        
        # Ensure integer columns stay integers (like ports)
        int_cols = ['Destination Port', 'Total Fwd Packets', 'Total Backward Packets']
        for col in int_cols:
            if col in samples.columns:
                samples[col] = samples[col].round().astype(int)

        return samples

    def start_simulation(self, network_id, traffic_type, interval=1.0):
        """Start a simulation thread for a specific network"""
        if network_id in self.active_simulations and self.active_simulations[network_id]['running']:
            logger.info(f"Simulation already running for Network {network_id}")
            return

        self.active_simulations[network_id] = {
            'running': True,
            'type': traffic_type,
            'thread': None,
            'latest_result': None,
            'stats': {'total': 0, 'attacks': 0}
        }

        def run_sim():
            logger.info(f"Started simulation for Network {network_id} ({traffic_type})")
            print(f"DEBUG: Simulation thread started for {network_id}")
            while self.active_simulations[network_id]['running']:
                try:
                    # Generate data
                    df = self.generate_traffic(traffic_type, batch_size=1)
                    
                    # Direct memory prediction
                    result, _ = self.prediction_service.predict_from_dataframe(df)
                    
                    # Update stats
                    self.active_simulations[network_id]['latest_result'] = result
                    self.active_simulations[network_id]['stats']['total'] += 1
                    
                    # Count attacks
                    if result and 'predictions' in result:
                        pred_class = result['predictions'][0]
                        # print(f"DEBUG: Network {network_id} prediction: {pred_class}")
                        if pred_class != 'BENIGN':
                             self.active_simulations[network_id]['stats']['attacks'] += 1
                    
                    time.sleep(interval)
                    
                except Exception as e:
                    logger.error(f"Error in simulation {network_id}: {e}")
                    print(f"DEBUG: Exception in simulation {network_id}: {e}")
                    import traceback
                    traceback.print_exc()
                    time.sleep(1)
        
        print(f"DEBUG: Launching thread for {network_id}")

        thread = threading.Thread(target=run_sim)
        thread.daemon = True
        self.active_simulations[network_id]['thread'] = thread
        thread.start()

    def stop_simulation(self, network_id):
        """Stop simulation for a specific network"""
        if network_id in self.active_simulations:
            self.active_simulations[network_id]['running'] = False
            logger.info(f"Stopped simulation for Network {network_id}")

    def get_status(self):
        """Get status of all simulations"""
        status = {}
        for net_id, sim in self.active_simulations.items():
            status[net_id] = {
                'running': sim['running'],
                'type': sim['type'],
                'stats': sim['stats'],
                'latest_result': sim['latest_result']
            }
        return status
