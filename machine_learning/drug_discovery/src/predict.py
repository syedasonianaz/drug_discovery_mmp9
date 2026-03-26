import joblib
import numpy as np
import os
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

class DrugPredictor:
    def __init__(self, model_name: str):
        # 1. Load the saved model
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(base_path, "models", model_name)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
            
        self.model = joblib.load(model_path)
        print(f"--- Model '{model_name}' loaded successfully ---")

    def _smiles_to_fp(self, smiles: str):
        """Internal helper to convert SMILES to 2048-bit fingerprint."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        arr = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr.reshape(1, -1)

    def predict(self, smiles: str):
        """Predicts the pIC50 value for a given SMILES string."""
        fp = self._smiles_to_fp(smiles)
        if fp is None:
            return "Invalid SMILES string."
        
        prediction = self.model.predict(fp)[0]
        return round(float(prediction), 3)

if __name__ == "__main__":
    # Test with a known potent molecule for MMP9 or a random one
    # Aspirin SMILES for example: CC(=O)OC1=CC=CC=C1C(=O)O
    predictor = DrugPredictor("mmp9_rf_model.joblib")
    
    test_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O" 
    result = predictor.predict(test_smiles)
    
    print(f"\nSMILES: {test_smiles}")
    print(f"Predicted pIC50: {result}")