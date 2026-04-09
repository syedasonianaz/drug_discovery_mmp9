import pandas as pd
import numpy as np
import os
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


class MoleculePreprocessor:
    """
    Handles scaffold splitting and Morgan fingerprint generation.
    Class imbalance is handled via class_weight='balanced' in training — no SMOTE.
    SMILES strings are saved alongside feature matrices for warhead analysis in training.
    """

    def __init__(self, radius=2, n_bits=2048):
        self.radius = radius
        self.n_bits = n_bits

    def _get_scaffold(self, smiles):
        """Generates Murcko scaffold SMILES for a given compound."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)

    def _generate_fp(self, smiles):
        """Generates Morgan fingerprint bit vector as numpy array."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros((self.n_bits,))
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, self.radius, nBits=self.n_bits)
        return np.array(fp)

    def scaffold_split(self, df, train_size=0.8):
        """
        Groups compounds by Murcko scaffold and splits into train/test.
        Ensures zero scaffold overlap between splits (no structural leakage).
        """
        print("Calculating Murcko scaffolds for all compounds...")
        df = df.copy()
        df['scaffold'] = df['canonical_smiles'].apply(self._get_scaffold)

        scaffold_to_indices = defaultdict(list)
        for idx, scaffold in enumerate(df['scaffold']):
            scaffold_to_indices[scaffold].append(idx)

        # Sort largest scaffolds first for deterministic splitting
        scaffold_sets = sorted(scaffold_to_indices.values(), key=len, reverse=True)

        print(f"Splitting data into {int(train_size*100)}/{int((1-train_size)*100)} train/test sets based on scaffolds...")
        train_indices, test_indices = [], []
        for scaffold_indices in scaffold_sets:
            if len(train_indices) / len(df) < train_size:
                train_indices.extend(scaffold_indices)
            else:
                test_indices.extend(scaffold_indices)

        df_train = df.iloc[train_indices].copy()
        df_test = df.iloc[test_indices].copy()

        print(f"Train set size: {len(df_train)} | Test set size: {len(df_test)}")
        print(f"Train active %: {df_train['bioactivity_class'].mean()*100:.1f}%")
        print(f"Test active %:  {df_test['bioactivity_class'].mean()*100:.1f}%")

        # Scaffold overlap check
        overlap = set(df_train['scaffold']) & set(df_test['scaffold'])
        print(f"Scaffold overlap between train and test: {len(overlap)}")
        if len(overlap) == 0:
            print("Success: Train and Test sets have mutually exclusive scaffolds.")
        else:
            print("Warning: Scaffold overlap detected — data leakage possible.")

        return df_train, df_test

    def prepare_matrices(self, df_train, df_test):
        """
        Converts dataframes to Morgan fingerprint matrices and label arrays.
        No resampling applied — class_weight='balanced' in RandomForest handles imbalance.
        Returns SMILES arrays alongside X/y for downstream warhead analysis.
        """
        print("\nGenerating Morgan fingerprints (Radius=2, 2048 bits) for training data...")
        X_train = np.stack(df_train['canonical_smiles'].apply(self._generate_fp))
        y_train = df_train['bioactivity_class'].values
        smiles_train = df_train['canonical_smiles'].values

        print("Generating Morgan fingerprints for testing data...")
        X_test = np.stack(df_test['canonical_smiles'].apply(self._generate_fp))
        y_test = df_test['bioactivity_class'].values
        smiles_test = df_test['canonical_smiles'].values

        print(f"Feature extraction complete. X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

        return X_train, y_train, X_test, y_test, smiles_train, smiles_test

    def save_processed_data(self, X_train, y_train, X_test, y_test,
                            smiles_train, smiles_test, output_path):
        """
        Saves feature matrices, labels, and SMILES strings into a single compressed npz.
        SMILES are required by train.py for warhead bias analysis.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.savez_compressed(
            output_path,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            smiles_train=smiles_train,
            smiles_test=smiles_test
        )
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"\nModel-ready data saved to {output_path}")
        print(f"File size: {file_size_mb:.2f} MB")


if __name__ == "__main__":
    df = pd.read_csv('../data/processed/mmp9_clean.csv')

    preprocessor = MoleculePreprocessor()
    df_train, df_test = preprocessor.scaffold_split(df)
    X_tr, y_tr, X_te, y_te, smi_tr, smi_te = preprocessor.prepare_matrices(df_train, df_test)
    preprocessor.save_processed_data(
        X_tr, y_tr, X_te, y_te, smi_tr, smi_te,
        '../data/processed/mmp9_model_ready_splits.npz'
    )
