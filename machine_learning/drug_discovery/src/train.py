import pandas as pd
import numpy as np
import os
import joblib
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

class ModelTrainer:
    def __init__(self, data_path: str):
        self.df = pd.read_csv(data_path)
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)

    def generate_fingerprints(self, smiles_column='canonical_smiles'):
        """Converts SMILES into a 2048-bit Morgan Fingerprint array."""
        print("Generating Morgan Fingerprints...")
        mols = [Chem.MolFromSmiles(s) for s in self.df[smiles_column]]
        fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) for m in mols]
        
        # Convert fingerprints to a numpy array
        np_fps = []
        for fp in fps:
            arr = np.zeros((1,))
            DataStructs.ConvertToNumpyArray(fp, arr)
            np_fps.append(arr)
        
        return np.array(np_fps)

    def run_training(self):
        X = self.generate_fingerprints()
        y = self.df['pIC50'].values

        # Split data (80% Train, 20% Test)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        print(f"Training on {len(X_train)} molecules...")
        self.model.fit(X_train, y_train)

        # Evaluate
        predictions = self.model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        print(f"✅ Training Complete!")
        print(f"📊 R² Score: {r2:.3f}")
        print(f"📊 MSE: {mse:.3f}")
        
        return self

    def save_model(self, model_name: str):
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(base_path, "models", model_name)
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        joblib.dump(self.model, model_path)
        print(f"💾 Model saved to: {model_path}")

if __name__ == "__main__":
    processed_data = r"E:\VS Code\machine_learning\drug_discovery\data\processed\mmp9_processed.csv"
    
    trainer = ModelTrainer(processed_data)
    trainer.run_training().save_model("mmp9_rf_model.joblib")