import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import matthews_corrcoef, confusion_matrix

class ModelTrainer:
    """
    Handles training the champion Random Forest model and 
    serializing it for deployment.
    """

    def __init__(self, n_estimators=300, max_depth=40, threshold=0.4417):
        # We hardcode our optimized "Champion" parameters as defaults
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.threshold = threshold
        self.model = None

    def load_data(self, data_path):
        """Loads the .npz archive created by preprocess.py."""
        print(f"Loading matrices from {data_path}...")
        data = np.load(data_path)
        return data['X_train'], data['y_train'], data['X_test'], data['y_test']

    def train(self, X_train, y_train):
        """Trains the Random Forest with our optimized settings."""
        print("Training Champion Random Forest... (this may take a moment)")
        
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=2,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train, y_train)
        print("Training complete.")

    def evaluate(self, X_test, y_test):
        """Runs evaluation using the mathematical custom threshold."""
        if not self.model:
            raise ValueError("Model not trained yet!")

        # Get probabilities and apply custom threshold
        probs = self.model.predict_proba(X_test)[:, 1]
        preds = (probs >= self.threshold).astype(int)
        
        mcc = matthews_corrcoef(y_test, preds)
        print(f"\n--- Evaluation Results ---")
        print(f"Custom Threshold: {self.threshold}")
        print(f"Matthews Correlation Coefficient: {mcc:.4f}")
        print(f"Confusion Matrix:\n{confusion_matrix(y_test, preds)}")
        return mcc

    def save_model(self, model_dir='../models/'):
        """Saves the model and metadata for the virtual screening script."""
        if not self.model:
            print("No model to save.")
            return

        os.makedirs(model_dir, exist_ok=True)
        
        # Save the model object
        model_path = os.path.join(model_dir, 'mmp9_rf_champion.pkl')
        joblib.dump(self.model, model_path)
        
        # Save metadata (the 'key' to using the model correctly)
        meta_path = os.path.join(model_dir, 'mmp9_rf_metadata.pkl')
        metadata = {
            'optimal_threshold': self.threshold,
            'model_type': 'RandomForestClassifier',
            'n_features': 2048,
            'target': 'MMP-9'
        }
        joblib.dump(metadata, meta_path)
        
        print(f"Model and Metadata saved to {model_dir}")

# --- Automation Script ---
if __name__ == "__main__":
    # 1. Initialize Trainer
    trainer = ModelTrainer()
    
    # 2. Load the ready-to-use data from preprocess.py
    X_train, y_train, X_test, y_test = trainer.load_data('../data/processed/mmp9_model_ready.npz')
    
    # 3. Execute Pipeline
    trainer.train(X_train, y_train)
    trainer.evaluate(X_test, y_test)
    trainer.save_model()