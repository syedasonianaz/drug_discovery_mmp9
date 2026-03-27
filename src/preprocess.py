import pandas as pd
import numpy as np
import os
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict
from imblearn.over_sampling import SMOTE

class MoleculePreprocessor:
    """
    Handles the structural splitting, feature extraction (Morgan FPs),
    and class balancing (SMOTE) for the bioactivity dataset.
    """
    
    def __init__(self, radius=2, n_bits=2048):
        self.radius = radius
        self.n_bits = n_bits
        self.smote = SMOTE(random_state=42)

    def _get_scaffold(self, smiles):
        """Internal helper to generate Murcko Scaffold."""
        mol = Chem.MolFromSmiles(smiles)
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False) if mol else None

    def _generate_fp(self, smiles):
        """Internal helper to generate Morgan Fingerprint bit vector."""
        mol = Chem.MolFromSmiles(smiles)
        if not mol: return np.zeros((self.n_bits,))
        return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, self.radius, nBits=self.n_bits))

    def scaffold_split(self, df, train_size=0.8):
        """
        Groups compounds by scaffold and splits them into train/test sets
        to ensure no structural leakage.
        """
        print("Performing Scaffold Split...")
        df['scaffold'] = df['canonical_smiles'].apply(self._get_scaffold)
        
        scaffold_to_indices = defaultdict(list)
        for idx, scaffold in enumerate(df['scaffold']):
            scaffold_to_indices[scaffold].append(idx)
            
        scaffold_sets = sorted(scaffold_to_indices.values(), key=len, reverse=True)
        
        train_indices, test_indices = [], []
        for indices in scaffold_sets:
            if len(train_indices) / len(df) < train_size:
                train_indices.extend(indices)
            else:
                test_indices.extend(indices)
                
        return df.iloc[train_indices].copy(), df.iloc[test_indices].copy()

    def prepare_matrices(self, df_train, df_test):
        """
        Converts dataframes to feature matrices (X) and labels (y),
        then applies SMOTE to the training set.
        """
        print("Generating Morgan Fingerprints...")
        X_train = np.stack(df_train['canonical_smiles'].apply(self._generate_fp))
        y_train = df_train['bioactivity_class'].values
        
        X_test = np.stack(df_test['canonical_smiles'].apply(self._generate_fp))
        y_test = df_test['bioactivity_class'].values
        
        print("Applying SMOTE to balance training data...")
        X_train_bal, y_train_bal = self.smote.fit_resample(X_train, y_train)
        
        return X_train_bal, y_train_bal, X_test, y_test

    def save_processed_data(self, X_train, y_train, X_test, y_test, output_path):
        """Saves the final matrices as a compressed numpy archive."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.savez_compressed(output_path, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
        print(f"Model-ready data saved to {output_path}")

# --- Execution Logic ---
if __name__ == "__main__":
    # 1. Load the cleaned data from ingest.py
    df = pd.read_csv('../data/processed/mmp9_clean.csv')
    
    # 2. Preprocess
    preprocessor = MoleculePreprocessor()
    df_train, df_test = preprocessor.scaffold_split(df)
    
    # 3. Featurize and Balance
    X_tr, y_tr, X_te, y_te = preprocessor.prepare_matrices(df_train, df_test)
    
    # 4. Export
    preprocessor.save_processed_data(X_tr, y_tr, X_te, y_te, '../data/processed/mmp9_model_ready.npz')