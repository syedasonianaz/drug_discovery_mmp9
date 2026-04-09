import joblib
import numpy as np
import os

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, MolFromSmarts

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Hydroxamic acid SMARTS — loaded once at module level
HYDROXAMIC_SMARTS = MolFromSmarts('[C](=O)[NH][OH]')


class DrugPredictor:
    """
    Loads the champion RF model and metadata at startup.
    Converts SMILES to Morgan fingerprint, predicts activity,
    and explicitly flags hydroxamic acid warhead presence.

    Every prediction response includes:
      - prediction:    'active' or 'inactive'
      - probability:   confidence score (0.0 - 1.0)
      - warhead_flag:  True/False — hydroxamic acid detected
      - warhead_note:  explanation of model bias if warhead present
    """

    def __init__(self, model_name: str = 'mmp9_rf_champion.pkl'):
        base_path  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(base_path, 'models', model_name)
        meta_path  = os.path.join(base_path, 'models', 'mmp9_rf_metadata.pkl')

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata not found at {meta_path}")

        self.model    = joblib.load(model_path)
        self.metadata = joblib.load(meta_path)
        self.threshold = self.metadata['optimal_threshold']

        print(f"Model '{model_name}' loaded. Threshold: {self.threshold:.4f}")

    def _smiles_to_fp(self, smiles: str):
        """Converts SMILES string to 2048-bit Morgan fingerprint numpy array."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp  = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        arr = np.zeros((2048,))
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr.reshape(1, -1)

    def _check_warhead(self, smiles: str) -> bool:
        """Returns True if SMILES contains hydroxamic acid warhead."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        return mol.HasSubstructMatch(HYDROXAMIC_SMARTS)

    def predict(self, smiles: str) -> dict:
        """
        Predicts MMP-9 inhibitory activity for a given SMILES string.

        Returns a dict with:
          prediction   - 'active' or 'inactive'
          probability  - model confidence (probability of active class)
          warhead_flag - True if hydroxamic acid detected
          warhead_note - bias warning if warhead present, else None
          error        - error message if SMILES invalid, else None
        """
        # Validate SMILES
        fp = self._smiles_to_fp(smiles)
        if fp is None:
            return {
                'prediction':   None,
                'probability':  None,
                'warhead_flag': None,
                'warhead_note': None,
                'error':        'Invalid SMILES string. Please provide a valid structure.'
            }

        # Predict
        prob       = self.model.predict_proba(fp)[0][1]
        prediction = 'active' if prob >= self.threshold else 'inactive'

        # Warhead check
        warhead = self._check_warhead(smiles)
        warhead_note = None
        if warhead:
            warhead_note = (
                "This compound contains a hydroxamic acid warhead — "
                "the model has a known bias toward predicting warhead-carrying "
                "compounds as active (TN=0 on warhead-containing inactives). "
                "Interpret this prediction with caution."
            )

        return {
            'prediction':   prediction,
            'probability':  round(float(prob), 4),
            'warhead_flag': warhead,
            'warhead_note': warhead_note,
            'error':        None
        }


if __name__ == "__main__":
    predictor = DrugPredictor()

    # Test with a hydroxamic acid MMP inhibitor (should be active + warhead flagged)
    test_cases = [
        ("CC(=O)OC1=CC=CC=C1C(=O)O", "Aspirin — not an MMP inhibitor"),
        ("O=C(NO)c1ccc(NC(=O)c2cccc(F)c2)cc1", "Hydroxamic acid compound"),
    ]

    for smiles, label in test_cases:
        result = predictor.predict(smiles)
        print(f"\n{label}")
        print(f"  SMILES:       {smiles}")
        print(f"  Prediction:   {result['prediction']}")
        print(f"  Probability:  {result['probability']}")
        print(f"  Warhead flag: {result['warhead_flag']}")
        if result['warhead_note']:
            print(f"  Note:         {result['warhead_note']}")
        if result['error']:
            print(f"  Error:        {result['error']}")
