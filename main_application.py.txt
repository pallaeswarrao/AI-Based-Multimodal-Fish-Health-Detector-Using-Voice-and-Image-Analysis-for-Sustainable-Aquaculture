"""
Main Application for Fish Health Detector
This module provides the main interface for the fish health detection system.
"""

import os
import sys
import numpy as np
import pandas as pd
import cv2
import librosa
from typing import Dict, List, Tuple, Optional
import json
import datetime
from pathlib import Path

# Import custom modules
from data_preprocessor import ImagePreprocessor, AudioPreprocessor, DataAugmentor
from feature_extractor import MultimodalFeatureExtractor
from ml_models import TraditionalMLModels, DeepLearningModels, EnsembleModel, ModelEvaluator

class FishHealthDetector:
    """Main class for fish health detection system."""
    
    def __init__(self, config_path: str = None):
        """
        Initialize the fish health detector.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        
        # Initialize components
        self.image_preprocessor = ImagePreprocessor(
            target_size=self.config.get('image_size', (224, 224))
        )
        self.audio_preprocessor = AudioPreprocessor(
            sample_rate=self.config.get('sample_rate', 22050),
            duration=self.config.get('audio_duration', 5.0)
        )
        self.feature_extractor = MultimodalFeatureExtractor()
        self.data_augmentor = DataAugmentor()
        
        # Initialize models
        self.traditional_models = TraditionalMLModels()
        self.deep_models = DeepLearningModels()
        self.ensemble_model = EnsembleModel()
        
        # Health status mapping
        self.health_status = {
            0: 'Healthy',
            1: 'Stressed',
            2: 'Diseased'
        }
        
        # Initialize data storage
        self.detection_history = []
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from file or use defaults."""
        default_config = {
            'image_size': (224, 224),
            'sample_rate': 22050,
            'audio_duration': 5.0,
            'confidence_threshold': 0.7,
            'alert_threshold': 0.8,
            'model_save_path': 'models/',
            'data_save_path': 'data/',
            'results_save_path': 'results/'
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = json.load(f)
            default_config.update(user_config)
        
        return default_config
    
    def process_single_sample(self, image_path: str, audio_path: str) -> Dict:
        """
        Process a single fish sample (image + audio).
        
        Args:
            image_path: Path to fish image
            audio_path: Path to audio recording
            
        Returns:
            Detection results
        """
        try:
            # Preprocess data
            image = self.image_preprocessor.preprocess_image(image_path)
            audio = self.audio_preprocessor.preprocess_audio(audio_path)
            
            # Extract features
            features = self.feature_extractor.extract_combined_features(image, audio)
            
            # Convert features to array
            feature_array = np.array(list(features.values())).reshape(1, -1)
            
            # Make prediction using ensemble
            if self.traditional_models.is_fitted:
                prediction = self.traditional_models.predict(feature_array, 'random_forest')[0]
                probabilities = self.traditional_models.predict_proba(feature_array, 'random_forest')[0]
                confidence = np.max(probabilities)
            else:
                # Default prediction if models not trained
                prediction = 0
                confidence = 0.5
                probabilities = [0.33, 0.33, 0.34]
            
            # Create result
            result = {
                'timestamp': datetime.datetime.now().isoformat(),
                'image_path': image_path,
                'audio_path': audio_path,
                'prediction': int(prediction),
                'health_status': self.health_status[int(prediction)],
                'confidence': float(confidence),
                'probabilities': {
                    'healthy': float(probabilities[0]),
                    'stressed': float(probabilities[1]),
                    'diseased': float(probabilities[2])
                },
                'features': features,
                'alert_triggered': confidence > self.config['alert_threshold'] and prediction > 0
            }
            
            # Store in history
            self.detection_history.append(result)
            
            return result
            
        except Exception as e:
            return {
                'error': str(e),
                'timestamp': datetime.datetime.now().isoformat(),
                'image_path': image_path,
                'audio_path': audio_path
            }
    
    def batch_process(self, data_directory: str) -> List[Dict]:
        """
        Process multiple samples from a directory.
        
        Args:
            data_directory: Directory containing image and audio files
            
        Returns:
            List of detection results
        """
        results = []
        data_path = Path(data_directory)
        
        # Find image and audio file pairs
        image_files = list(data_path.glob('*.jpg')) + list(data_path.glob('*.png'))
        audio_files = list(data_path.glob('*.wav')) + list(data_path.glob('*.mp3'))
        
        # Match files by name
        for image_file in image_files:
            image_name = image_file.stem
            matching_audio = None
            
            for audio_file in audio_files:
                if audio_file.stem == image_name:
                    matching_audio = audio_file
                    break
            
            if matching_audio:
                result = self.process_single_sample(str(image_file), str(matching_audio))
                results.append(result)
                print(f"Processed: {image_name}")
            else:
                print(f"No matching audio found for: {image_name}")
        
        return results
    
    def train_system(self, training_data_path: str, labels_file: str) -> Dict:
        """
        Train the fish health detection system.
        
        Args:
            training_data_path: Path to training data directory
            labels_file: Path to CSV file with labels
            
        Returns:
            Training results
        """
        print("Starting system training...")
        
        # Load labels
        labels_df = pd.read_csv(labels_file)
        
        # Prepare training data
        X_features = []
        y_labels = []
        
        for _, row in labels_df.iterrows():
            image_path = os.path.join(training_data_path, row['image_file'])
            audio_path = os.path.join(training_data_path, row['audio_file'])
            label = row['health_status']
            
            if os.path.exists(image_path) and os.path.exists(audio_path):
                try:
                    # Preprocess data
                    image = self.image_preprocessor.preprocess_image(image_path)
                    audio = self.audio_preprocessor.preprocess_audio(audio_path)
                    
                    # Extract features
                    features = self.feature_extractor.extract_combined_features(image, audio)
                    
                    X_features.append(list(features.values()))
                    y_labels.append(label)
                    
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")
                    continue
        
        if not X_features:
            raise ValueError("No valid training samples found")
        
        # Convert to numpy arrays
        X = np.array(X_features)
        y = np.array(y_labels)
        
        print(f"Training with {len(X)} samples...")
        
        # Train traditional models
        training_results = self.traditional_models.train_models(X, y)
        
        # Save models
        model_save_path = self.config['model_save_path']
        os.makedirs(model_save_path, exist_ok=True)
        self.traditional_models.save_models(model_save_path)
        
        print("Training completed!")
        return training_results
    
    def load_trained_models(self, model_path: str = None):
        """Load pre-trained models."""
        if model_path is None:
            model_path = self.config['model_save_path']
        
        if os.path.exists(model_path):
            self.traditional_models.load_models(model_path)
            print("Models loaded successfully!")
        else:
            print(f"Model path {model_path} not found!")
    
    def generate_report(self, results: List[Dict], save_path: str = None) -> str:
        """
        Generate a comprehensive report from detection results.
        
        Args:
            results: List of detection results
            save_path: Path to save the report
            
        Returns:
            Report content as string
        """
        if not results:
            return "No results to report."
        
        # Calculate statistics
        total_samples = len(results)
        healthy_count = sum(1 for r in results if r.get('prediction') == 0)
        stressed_count = sum(1 for r in results if r.get('prediction') == 1)
        diseased_count = sum(1 for r in results if r.get('prediction') == 2)
        alerts_triggered = sum(1 for r in results if r.get('alert_triggered', False))
        
        # Calculate average confidence
        confidences = [r.get('confidence', 0) for r in results if 'confidence' in r]
        avg_confidence = np.mean(confidences) if confidences else 0
        
        # Generate report
        report = f"""
Fish Health Detection Report
Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

SUMMARY STATISTICS:
==================
Total Samples Processed: {total_samples}
Healthy Fish: {healthy_count} ({healthy_count/total_samples*100:.1f}%)
Stressed Fish: {stressed_count} ({stressed_count/total_samples*100:.1f}%)
Diseased Fish: {diseased_count} ({diseased_count/total_samples*100:.1f}%)
Alerts Triggered: {alerts_triggered}
Average Confidence: {avg_confidence:.3f}

DETAILED RESULTS:
================
"""
        
        for i, result in enumerate(results[:10]):  # Show first 10 results
            if 'error' not in result:
                report += f"""
Sample {i+1}:
  Status: {result.get('health_status', 'Unknown')}
  Confidence: {result.get('confidence', 0):.3f}
  Alert: {'Yes' if result.get('alert_triggered') else 'No'}
  Timestamp: {result.get('timestamp', 'Unknown')}
"""
        
        if len(results) > 10:
            report += f"\n... and {len(results) - 10} more samples\n"
        
        # Save report if path provided
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            print(f"Report saved to: {save_path}")
        
        return report
    
    def real_time_monitoring(self, camera_id: int = 0, audio_device: int = 0, 
                           duration: int = 60):
        """
        Perform real-time monitoring (simulation).
        
        Args:
            camera_id: Camera device ID
            audio_device: Audio device ID
            duration: Monitoring duration in seconds
        """
        print(f"Starting real-time monitoring for {duration} seconds...")
        print("Note: This is a simulation - actual implementation would require hardware integration")
        
        # Simulate real-time monitoring
        import time
        import random
        
        start_time = time.time()
        detection_count = 0
        
        while time.time() - start_time < duration:
            # Simulate detection
            detection_count += 1
            
            # Generate random prediction for simulation
            prediction = random.choice([0, 1, 2])
            confidence = random.uniform(0.5, 0.95)
            
            result = {
                'timestamp': datetime.datetime.now().isoformat(),
                'detection_id': detection_count,
                'prediction': prediction,
                'health_status': self.health_status[prediction],
                'confidence': confidence,
                'alert_triggered': confidence > self.config['alert_threshold'] and prediction > 0
            }
            
            print(f"Detection {detection_count}: {result['health_status']} (Confidence: {confidence:.3f})")
            
            if result['alert_triggered']:
                print("⚠️  ALERT: Potential health issue detected!")
            
            self.detection_history.append(result)
            
            # Wait before next detection
            time.sleep(5)
        
        print(f"Monitoring completed. Total detections: {detection_count}")
    
    def export_data(self, export_path: str, format: str = 'csv'):
        """
        Export detection history to file.
        
        Args:
            export_path: Path to save exported data
            format: Export format ('csv' or 'json')
        """
        if not self.detection_history:
            print("No data to export.")
            return
        
        if format.lower() == 'csv':
            # Convert to DataFrame and save as CSV
            df_data = []
            for record in self.detection_history:
                if 'error' not in record:
                    row = {
                        'timestamp': record.get('timestamp'),
                        'health_status': record.get('health_status'),
                        'confidence': record.get('confidence'),
                        'alert_triggered': record.get('alert_triggered')
                    }
                    df_data.append(row)
            
            df = pd.DataFrame(df_data)
            df.to_csv(export_path, index=False)
            
        elif format.lower() == 'json':
            with open(export_path, 'w') as f:
                json.dump(self.detection_history, f, indent=2)
        
        print(f"Data exported to: {export_path}")

def main():
    """Main function to demonstrate the fish health detector."""
    print("Fish Health Detector - AI-Based Multimodal System")
    print("=" * 50)
    
    # Initialize detector
    detector = FishHealthDetector()
    
    # Create sample configuration
    config = {
        'image_size': (224, 224),
        'sample_rate': 22050,
        'audio_duration': 5.0,
        'confidence_threshold': 0.7,
        'alert_threshold': 0.8
    }
    
    # Save configuration
    with open('/home/ubuntu/fish_health_detector/config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("System initialized successfully!")
    print("\nAvailable functions:")
    print("1. Process single sample")
    print("2. Batch processing")
    print("3. Train system")
    print("4. Real-time monitoring simulation")
    print("5. Generate report")
    
    # Demonstrate real-time monitoring simulation
    print("\nRunning real-time monitoring simulation...")
    detector.real_time_monitoring(duration=30)  # 30 seconds simulation
    
    # Generate and save report
    report = detector.generate_report(detector.detection_history)
    print("\n" + report)
    
    # Export data
    detector.export_data('/home/ubuntu/fish_health_detector/results/detection_results.csv', 'csv')
    detector.export_data('/home/ubuntu/fish_health_detector/results/detection_results.json', 'json')
    
    print("\nDemo completed successfully!")

if __name__ == "__main__":
    main()

