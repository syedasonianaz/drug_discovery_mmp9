import pandas as pd
import numpy as np
import joblib
import os
from chembl_webresource_client.new_client import new_client
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

class VirtualScreener:
    """
    Deploys a trained model to screen chemical libraries for 
    potential bioactivity. Optimized for ChEMBL drug repurposing.
    """
    
    def __init__(self, model_path, meta_path):
        # Load the brain (model) and the instructions (metadata)
        self.model = joblib.load(model_path)
        metadata = joblib.load(meta_path)
        
        self.threshold = metadata['optimal_threshold']
        self.n_bits = metadata.get('n_features', 2048)
        self.radius = metadata.get('fingerprint_radius', 2)
        
        print(f"Screener initialized with threshold: {self.threshold:.4f}")

    def fetch_chembl_approved(self, limit=None):
        """Downloads approved drugs (Phase 4) from ChEMBL."""
        print("Fetching approved drugs from ChEMBL API...")
        molecule = new_client.molecule
        approved = molecule.filter(max_phase=4)
        
        drug_list = []
        for drug in approved:
            if drug.get('molecule_structures') and drug['molecule_structures'].get('canonical_smiles'):
                drug_list.append({
                    'chembl_id': drug['molecule_chembl_id'],
                    'name': drug.get('pref_name', 'Unknown'),
                    'smiles': drug['molecule_structures']['canonical_smiles']
                })
            if limit and len(drug_list) >= limit: break
                
        self.library = pd.DataFrame(drug_list).drop_duplicates('smiles')
        print(f"Library loaded with {len(self.library)} unique drugs.")
        return self.library

    def _apply_filters(self, smiles):
        """Internal Lipinski's Rule of 5 filter."""
        mol = Chem.MolFromSmiles(smiles)
        if not mol: return False
        return (Descriptors.ExactMolWt(mol) <= 500 and 
                Descriptors.MolLogP(mol) <= 5 and 
                Descriptors.NumHDonors(mol) <= 5 and 
                Descriptors.NumHAcceptors(mol) <= 10)

    def _generate_fp(self, smiles):
        """Internal fingerprint generator."""
        mol = Chem.MolFromSmiles(smiles)
        if not mol: return np.zeros((self.n_bits,))
        return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, self.radius, nBits=self.n_bits))

    def run_screen(self, output_path='../data/processed/screening_results.csv'):
        """Executes the full screening pipeline."""
        print("Filtering for drug-like properties (Lipinski)...")
        df = self.library[self.library['smiles'].apply(self._apply_filters)].copy()
        
        print("Featurizing survivors...")
        X = np.stack(df['smiles'].apply(self._generate_fp))
        
        print("Predicting probabilities...")
        probs = self.model.predict_proba(X)[:, 1]
        df['Active_Probability'] = probs
        df['is_hit'] = (probs >= self.threshold).astype(int)
        
        # Sort and save
        results = df.sort_values('Active_Probability', ascending=False)
        results.to_csv(output_path, index=False)
        print(f"Screening complete! Results saved to {output_path}")
        return results

if __name__ == "__main__":
    # Point to the models saved by training.py
    screener = VirtualScreener('../models/mmp9_rf_champion.pkl', '../models/mmp9_rf_metadata.pkl')
    screener.fetch_chembl_approved()
    hits = screener.run_screen()
    print(f"Top Hit: {hits.iloc[0]['name']} (Prob: {hits.iloc[0]['Active_Probability']:.4f})")