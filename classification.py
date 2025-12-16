import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os


class BirdSoundClassifier:
    def __init__(self, model_path=None):
        """
        Initialize the BirdSoundClassifier.
        
        Args:
            model_path (str): Path to a saved model file. If provided, loads the model.
        """
        self.model = None
        self.scaler = StandardScaler()
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def _create_model(self):
        """
        Create a new Random Forest model.
        
        Returns:
            sklearn model: The created model
        """
        return RandomForestClassifier(n_estimators=100, random_state=42)
    
    def train(self, features, labels, test_size=0.2):
        """
        Train the classifier.
        
        Args:
            features (numpy.ndarray): Feature matrix
            labels (numpy.ndarray): Target labels
            test_size (float): Proportion of data to use for testing
            
        Returns:
            dict: Training results including accuracy and classification report
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=test_size, random_state=42, stratify=labels
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Create and train model
        self.model = self._create_model()
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        results = {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': conf_matrix,
            'model': self.model
        }
        
        return results
    
    def predict(self, features):
        """
        Predict the class for given features.
        
        Args:
            features (numpy.ndarray): Feature matrix or single feature vector
            
        Returns:
            numpy.ndarray: Predicted class labels
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded. Call train() or load_model() first.")
        
        # Reshape if single sample
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Predict
        return self.model.predict(features_scaled)
    
    def predict_proba(self, features):
        """
        Predict class probabilities for given features.
        
        Args:
            features (numpy.ndarray): Feature matrix or single feature vector
            
        Returns:
            numpy.ndarray: Predicted class probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded. Call train() or load_model() first.")
        
        # Check if model supports predict_proba
        if not hasattr(self.model, 'predict_proba'):
            raise ValueError(f"Model {self.model_type} does not support probability predictions.")
        
        # Reshape if single sample
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Predict probabilities
        return self.model.predict_proba(features_scaled)
    
    def save_model(self, model_path, scaler_path=None):
        """
        Save the trained model and scaler.
        
        Args:
            model_path (str): Path to save the model
            scaler_path (str): Path to save the scaler. If None, uses model_path with '_scaler' suffix.
        """
        if self.model is None:
            raise ValueError("No trained model to save.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save model
        joblib.dump(self.model, model_path)
        
        # Save scaler
        if scaler_path is None:
            scaler_path = os.path.splitext(model_path)[0] + '_scaler.pkl'
        
        joblib.dump(self.scaler, scaler_path)
        
        print(f"Model saved to {model_path}")
        print(f"Scaler saved to {scaler_path}")
    
    def load_model(self, model_path, scaler_path=None):
        """
        Load a trained model and scaler.
        
        Args:
            model_path (str): Path to the saved model
            scaler_path (str): Path to the saved scaler. If None, tries to find a scaler with '_scaler' suffix.
        """
        # Load model
        self.model = joblib.load(model_path)
        
        # Determine scaler path if not provided
        if scaler_path is None:
            scaler_path = os.path.splitext(model_path)[0] + '_scaler.pkl'
        
        # Load scaler if it exists
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
        else:
            print(f"Warning: Scaler not found at {scaler_path}. Using default scaler.")
            self.scaler = StandardScaler()
        
        print(f"Model loaded from {model_path}")
        
    def get_feature_importance(self, feature_names=None):
        """
        Get feature importance (for models that support it).
        
        Args:
            feature_names (list): List of feature names
            
        Returns:
            pandas.DataFrame: DataFrame with feature importance
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded.")
        
        if not hasattr(self.model, 'feature_importances_'):
            raise ValueError(f"Model {self.model_type} does not support feature importance.")
        
        importances = self.model.feature_importances_
        
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(len(importances))]
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        return importance_df



