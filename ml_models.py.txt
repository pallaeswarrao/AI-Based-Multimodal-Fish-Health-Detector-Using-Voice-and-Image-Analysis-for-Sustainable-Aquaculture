"""
Machine Learning Models for Fish Health Detection
This module contains various ML models for classifying fish health status.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import joblib
import os

# Scikit-learn imports
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline

# TensorFlow/Keras imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50, VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class TraditionalMLModels:
    """Traditional machine learning models for fish health classification."""
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_fitted = False
        
    def prepare_models(self):
        """Initialize all traditional ML models."""
        self.models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            ),
            'svm': SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,
                random_state=42
            ),
            'logistic_regression': LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=42
            ),
            'mlp': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                alpha=0.001,
                max_iter=1000,
                random_state=42
            )
        }
    
    def train_models(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Train all traditional ML models.
        
        Args:
            X: Feature matrix
            y: Target labels
            
        Returns:
            Dictionary of model accuracies
        """
        if not self.models:
            self.prepare_models()
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        results = {}
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            results[name] = accuracy
            
            print(f"{name} accuracy: {accuracy:.4f}")
        
        self.is_fitted = True
        return results
    
    def hyperparameter_tuning(self, X: np.ndarray, y: np.ndarray, model_name: str) -> Dict:
        """
        Perform hyperparameter tuning for a specific model.
        
        Args:
            X: Feature matrix
            y: Target labels
            model_name: Name of the model to tune
            
        Returns:
            Best parameters found
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        # Encode labels and scale features
        y_encoded = self.label_encoder.fit_transform(y)
        X_scaled = self.scaler.fit_transform(X)
        
        # Define parameter grids
        param_grids = {
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10]
            },
            'gradient_boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 6, 9]
            },
            'svm': {
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto', 0.001, 0.01],
                'kernel': ['rbf', 'poly']
            }
        }
        
        if model_name not in param_grids:
            print(f"No parameter grid defined for {model_name}")
            return {}
        
        # Perform grid search
        grid_search = GridSearchCV(
            self.models[model_name],
            param_grids[model_name],
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_scaled, y_encoded)
        
        print(f"Best parameters for {model_name}: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        # Update model with best parameters
        self.models[model_name] = grid_search.best_estimator_
        
        return grid_search.best_params_
    
    def predict(self, X: np.ndarray, model_name: str = 'random_forest') -> np.ndarray:
        """
        Make predictions using a trained model.
        
        Args:
            X: Feature matrix
            model_name: Name of the model to use
            
        Returns:
            Predictions
        """
        if not self.is_fitted:
            raise ValueError("Models must be trained before making predictions")
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        X_scaled = self.scaler.transform(X)
        predictions = self.models[model_name].predict(X_scaled)
        
        # Decode labels
        return self.label_encoder.inverse_transform(predictions)
    
    def predict_proba(self, X: np.ndarray, model_name: str = 'random_forest') -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            X: Feature matrix
            model_name: Name of the model to use
            
        Returns:
            Prediction probabilities
        """
        if not self.is_fitted:
            raise ValueError("Models must be trained before making predictions")
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        X_scaled = self.scaler.transform(X)
        
        if hasattr(self.models[model_name], 'predict_proba'):
            return self.models[model_name].predict_proba(X_scaled)
        else:
            raise ValueError(f"Model {model_name} does not support probability predictions")
    
    def save_models(self, save_dir: str):
        """Save trained models to disk."""
        os.makedirs(save_dir, exist_ok=True)
        
        for name, model in self.models.items():
            joblib.dump(model, os.path.join(save_dir, f"{name}.pkl"))
        
        # Save scaler and label encoder
        joblib.dump(self.scaler, os.path.join(save_dir, "scaler.pkl"))
        joblib.dump(self.label_encoder, os.path.join(save_dir, "label_encoder.pkl"))
    
    def load_models(self, save_dir: str):
        """Load trained models from disk."""
        model_files = ['random_forest.pkl', 'gradient_boosting.pkl', 'svm.pkl', 
                      'logistic_regression.pkl', 'mlp.pkl']
        
        for model_file in model_files:
            model_path = os.path.join(save_dir, model_file)
            if os.path.exists(model_path):
                model_name = model_file.replace('.pkl', '')
                self.models[model_name] = joblib.load(model_path)
        
        # Load scaler and label encoder
        self.scaler = joblib.load(os.path.join(save_dir, "scaler.pkl"))
        self.label_encoder = joblib.load(os.path.join(save_dir, "label_encoder.pkl"))
        self.is_fitted = True

class DeepLearningModels:
    """Deep learning models for fish health classification."""
    
    def __init__(self, input_shape: Tuple[int, int, int] = (224, 224, 3), num_classes: int = 3):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.models = {}
        
    def create_cnn_model(self) -> keras.Model:
        """Create a custom CNN model for fish health classification."""
        model = keras.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.5),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def create_transfer_learning_model(self, base_model_name: str = 'resnet50') -> keras.Model:
        """
        Create a transfer learning model using pre-trained weights.
        
        Args:
            base_model_name: Name of the base model ('resnet50' or 'vgg16')
            
        Returns:
            Transfer learning model
        """
        if base_model_name.lower() == 'resnet50':
            base_model = ResNet50(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
        elif base_model_name.lower() == 'vgg16':
            base_model = VGG16(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
        else:
            raise ValueError(f"Unsupported base model: {base_model_name}")
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Add custom classification head
        model = keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.5),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def create_multimodal_model(self, image_features: int, audio_features: int) -> keras.Model:
        """
        Create a multimodal model that combines image and audio features.
        
        Args:
            image_features: Number of image features
            audio_features: Number of audio features
            
        Returns:
            Multimodal model
        """
        # Image branch
        image_input = layers.Input(shape=(image_features,), name='image_input')
        image_dense = layers.Dense(256, activation='relu')(image_input)
        image_dropout = layers.Dropout(0.3)(image_dense)
        image_output = layers.Dense(128, activation='relu')(image_dropout)
        
        # Audio branch
        audio_input = layers.Input(shape=(audio_features,), name='audio_input')
        audio_dense = layers.Dense(256, activation='relu')(audio_input)
        audio_dropout = layers.Dropout(0.3)(audio_dense)
        audio_output = layers.Dense(128, activation='relu')(audio_dropout)
        
        # Combine branches
        combined = layers.concatenate([image_output, audio_output])
        combined_dense = layers.Dense(256, activation='relu')(combined)
        combined_dropout = layers.Dropout(0.5)(combined_dense)
        final_output = layers.Dense(self.num_classes, activation='softmax')(combined_dropout)
        
        model = keras.Model(
            inputs=[image_input, audio_input],
            outputs=final_output
        )
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_model(self, model: keras.Model, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray, epochs: int = 50) -> keras.callbacks.History:
        """
        Train a deep learning model.
        
        Args:
            model: Keras model to train
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            epochs: Number of training epochs
            
        Returns:
            Training history
        """
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            )
        ]
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def evaluate_model(self, model: keras.Model, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate a trained model.
        
        Args:
            model: Trained Keras model
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Evaluation metrics
        """
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true_classes, y_pred_classes)
        
        # Get loss and accuracy from model evaluation
        loss, model_accuracy = model.evaluate(X_test, y_test, verbose=0)
        
        return {
            'accuracy': accuracy,
            'loss': loss,
            'model_accuracy': model_accuracy
        }

class EnsembleModel:
    """Ensemble model combining multiple classifiers."""
    
    def __init__(self):
        self.traditional_models = TraditionalMLModels()
        self.deep_models = {}
        self.weights = {}
        
    def train_ensemble(self, X_traditional: np.ndarray, X_deep: np.ndarray, 
                      y: np.ndarray) -> Dict[str, float]:
        """
        Train ensemble of traditional and deep learning models.
        
        Args:
            X_traditional: Features for traditional ML models
            X_deep: Features for deep learning models
            y: Target labels
            
        Returns:
            Performance metrics for each model
        """
        results = {}
        
        # Train traditional models
        traditional_results = self.traditional_models.train_models(X_traditional, y)
        results.update(traditional_results)
        
        # Calculate ensemble weights based on performance
        total_accuracy = sum(traditional_results.values())
        self.weights = {name: acc/total_accuracy for name, acc in traditional_results.items()}
        
        return results
    
    def predict_ensemble(self, X_traditional: np.ndarray, model_names: List[str] = None) -> np.ndarray:
        """
        Make ensemble predictions.
        
        Args:
            X_traditional: Features for traditional ML models
            model_names: List of model names to include in ensemble
            
        Returns:
            Ensemble predictions
        """
        if model_names is None:
            model_names = list(self.traditional_models.models.keys())
        
        predictions = []
        weights = []
        
        for name in model_names:
            if name in self.traditional_models.models:
                pred_proba = self.traditional_models.predict_proba(X_traditional, name)
                predictions.append(pred_proba)
                weights.append(self.weights.get(name, 1.0))
        
        if not predictions:
            raise ValueError("No valid models found for ensemble prediction")
        
        # Weighted average of predictions
        weights = np.array(weights) / np.sum(weights)
        ensemble_proba = np.average(predictions, axis=0, weights=weights)
        
        # Return class with highest probability
        return np.argmax(ensemble_proba, axis=1)

class ModelEvaluator:
    """Utility class for comprehensive model evaluation."""
    
    @staticmethod
    def evaluate_classification_performance(y_true: np.ndarray, y_pred: np.ndarray, 
                                          class_names: List[str] = None) -> Dict:
        """
        Comprehensive evaluation of classification performance.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Names of the classes
            
        Returns:
            Dictionary containing various performance metrics
        """
        if class_names is None:
            class_names = [f"Class_{i}" for i in range(len(np.unique(y_true)))]
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        
        # Classification report
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'class_names': class_names
        }
    
    @staticmethod
    def plot_training_history(history: keras.callbacks.History, save_path: str = None):
        """
        Plot training history for deep learning models.
        
        Args:
            history: Keras training history
            save_path: Path to save the plot
        """
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot accuracy
        ax1.plot(history.history['accuracy'], label='Training Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        
        # Plot loss
        ax2.plot(history.history['loss'], label='Training Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig

