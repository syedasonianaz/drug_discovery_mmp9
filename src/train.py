import numpy as np
import pandas as pd
import joblib
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import matthews_corrcoef, confusion_matrix
from sklearn.model_selection import cross_val_predict

from rdkit import Chem
from rdkit.Chem import MolFromSmarts

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Hydroxamic acid SMARTS — canonical MMP zinc-chelating warhead
HYDROXAMIC_SMARTS = MolFromSmarts('[C](=O)[NH][OH]')


def has_warhead(smiles):
    """Returns True if the SMILES contains a hydroxamic acid warhead."""
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return False
    return mol.HasSubstructMatch(HYDROXAMIC_SMARTS)


class ModelTrainer:
    """
    Trains the champion Random Forest model with automated threshold selection.
    Runs warhead bias analysis post-training.
    Saves model, metadata, and warhead-explicit model for deployment.
    """

    def __init__(self, n_estimators=300, max_depth=40):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.model = None
        self.optimal_threshold = None

    def load_data(self, data_path):
        """Loads npz archive produced by preprocess.py."""
        print(f"Loading matrices from {data_path}...")
        data = np.load(data_path, allow_pickle=True)
        X_train      = data['X_train']
        y_train      = data['y_train']
        X_test       = data['X_test']
        y_test       = data['y_test']
        smiles_train = data['smiles_train']
        smiles_test  = data['smiles_test']

        print(f"Train: {X_train.shape} features | {y_train.sum()} actives / {(y_train==0).sum()} inactives")
        print(f"Test:  {X_test.shape} features  | {y_test.sum()} actives / {(y_test==0).sum()} inactives")

        return X_train, y_train, X_test, y_test, smiles_train, smiles_test

    def train(self, X_train, y_train):
        """
        Trains champion Random Forest.
        Finds optimal decision threshold via 5-fold cross-validation on training set.
        Threshold is never selected using test data.
        """
        print("\nTraining Champion Random Forest...")
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=2,
            min_samples_leaf=2,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )

        # Automated threshold selection via CV — no test set involvement
        print("Calculating optimal decision threshold via 5-fold CV...")
        y_train_proba = cross_val_predict(
            self.model, X_train, y_train,
            cv=5, method='predict_proba', n_jobs=-1
        )[:, 1]

        thresholds = np.linspace(0.1, 0.9, 100)
        mcc_scores = [
            matthews_corrcoef(y_train, (y_train_proba >= t).astype(int))
            for t in thresholds
        ]

        self.optimal_threshold = thresholds[np.argmax(mcc_scores)]
        print(f"Optimal threshold: {self.optimal_threshold:.4f} (CV MCC: {max(mcc_scores):.4f})")

        # Train on full training set
        self.model.fit(X_train, y_train)
        print("Training complete.")

    def evaluate(self, X_test, y_test):
        """Evaluates champion model on test set using automated threshold."""
        if self.model is None:
            raise ValueError("Model not trained yet. Run train() first.")

        probs = self.model.predict_proba(X_test)[:, 1]
        preds = (probs >= self.optimal_threshold).astype(int)

        mcc = matthews_corrcoef(y_test, preds)
        cm  = confusion_matrix(y_test, preds)

        print(f"\n--- Champion Model Evaluation ---")
        print(f"Threshold: {self.optimal_threshold:.4f}")
        print(f"MCC: {mcc:.4f}")
        print(f"Confusion Matrix:")
        print(f"  TN: {cm[0,0]}  FP: {cm[0,1]}")
        print(f"  FN: {cm[1,0]}  TP: {cm[1,1]}")

        return mcc, preds

    def warhead_analysis(self, X_train, y_train, X_test, y_test,
                         smiles_train, smiles_test, y_pred_champ):
        """
        Analyses hydroxamic acid warhead bias in the model.
        Matches notebook logic exactly:
          - Prevalence across full dataset
          - MCC on warhead-free test subset
          - Confusion matrix on warhead-containing compounds only
          - Ablation A: train without warhead compounds
          - Ablation B: train with explicit warhead feature
        Returns warhead-explicit model (rf_hx) for deployment.
        """
        print("\n=== Hydroxamic Acid Warhead Analysis ===")

        all_smiles = np.concatenate([smiles_train, smiles_test])
        all_labels = np.concatenate([y_train, y_test])
        all_flags  = np.array([has_warhead(s) for s in all_smiles])

        active_with   = all_flags[all_labels == 1].sum()
        inactive_with = all_flags[all_labels == 0].sum()

        print(f"\nActives   with warhead: {active_with:4d} / {(all_labels==1).sum()} ({active_with/(all_labels==1).sum()*100:.1f}%)")
        print(f"Inactives with warhead: {inactive_with:4d} / {(all_labels==0).sum()} ({inactive_with/(all_labels==0).sum()*100:.1f}%)")

        # Warhead flags for test set
        test_flags      = np.array([has_warhead(s) for s in smiles_test])
        no_warhead_mask = ~test_flags

        mcc_all        = matthews_corrcoef(y_test, y_pred_champ)
        mcc_no_warhead = matthews_corrcoef(
            y_test[no_warhead_mask], y_pred_champ[no_warhead_mask]
        )

        print(f"\nMCC — full test set:          {mcc_all:.4f}")
        print(f"MCC — warhead-free compounds: {mcc_no_warhead:.4f}  (n={no_warhead_mask.sum()})")

        # Confusion matrix on warhead-containing compounds only
        cm_wh = confusion_matrix(y_test[test_flags], y_pred_champ[test_flags])
        print(f"\nConfusion Matrix — Warhead-Containing Compounds Only:")
        print(f"  TN: {cm_wh[0,0]}  FP: {cm_wh[0,1]}")
        print(f"  FN: {cm_wh[1,0]}  TP: {cm_wh[1,1]}")
        if cm_wh[0,0] == 0:
            print("  Finding: TN=0 — model cannot reject warhead-carrying inactives. Known blind spot.")

        # Ablation A: train without warhead compounds
        train_warhead    = np.array([has_warhead(s) for s in smiles_train])
        X_train_nohx     = X_train[~train_warhead]
        y_train_nohx     = y_train[~train_warhead]

        print(f"\nAblation A — Train size after removing warhead compounds: {len(X_train)} → {len(X_train_nohx)}")
        rf_nohx = RandomForestClassifier(
            n_estimators=500, class_weight='balanced', random_state=42, n_jobs=-1
        )
        rf_nohx.fit(X_train_nohx, y_train_nohx)

        # Ablation B: explicit warhead feature appended to fingerprint
        X_train_hx = np.hstack([X_train, train_warhead.astype(int).reshape(-1, 1)])
        X_test_hx  = np.hstack([X_test,  test_flags.astype(int).reshape(-1, 1)])

        print(f"Ablation B — New feature matrix shape: {X_train_hx.shape} (2048 bits + 1 warhead flag)")
        rf_hx = RandomForestClassifier(
            n_estimators=500, class_weight='balanced', random_state=42, n_jobs=-1
        )
        rf_hx.fit(X_train_hx, y_train)

        # Head-to-head comparison
        y_pred_nohx = rf_nohx.predict(X_test)
        y_pred_hx   = rf_hx.predict(X_test_hx)

        mcc_nohx = matthews_corrcoef(y_test, y_pred_nohx)
        mcc_hx   = matthews_corrcoef(y_test, y_pred_hx)
        cm_nohx  = confusion_matrix(y_test, y_pred_nohx)
        cm_hx    = confusion_matrix(y_test, y_pred_hx)
        cm_champ = confusion_matrix(y_test, y_pred_champ)

        print(f"\n{'='*58}")
        print(f"{'':22} {'rf_champ':>10} {'rf_nohx':>10} {'rf_hx':>10}")
        print(f"{'='*58}")
        print(f"{'MCC':22} {mcc_all:>10.4f} {mcc_nohx:>10.4f} {mcc_hx:>10.4f}")
        print(f"{'TN':22} {cm_champ[0,0]:>10} {cm_nohx[0,0]:>10} {cm_hx[0,0]:>10}")
        print(f"{'FP':22} {cm_champ[0,1]:>10} {cm_nohx[0,1]:>10} {cm_hx[0,1]:>10}")
        print(f"{'FN':22} {cm_champ[1,0]:>10} {cm_nohx[1,0]:>10} {cm_hx[1,0]:>10}")
        print(f"{'TP':22} {cm_champ[1,1]:>10} {cm_nohx[1,1]:>10} {cm_hx[1,1]:>10}")
        print(f"{'='*58}")

        return rf_hx

    def save_model(self, rf_hx, mcc_champ, model_dir='../models/'):
        """Saves champion model, warhead-explicit model, and metadata."""
        if self.model is None:
            print("No model to save.")
            return

        os.makedirs(model_dir, exist_ok=True)

        joblib.dump(self.model, os.path.join(model_dir, 'mmp9_rf_champion.pkl'))
        joblib.dump(rf_hx,      os.path.join(model_dir, 'mmp9_rf_explicit_warhead.pkl'))

        metadata = {
            'optimal_threshold':  self.optimal_threshold,
            'n_features':         2048,
            'fingerprint_radius': 2,
            'fingerprint_bits':   2048,
            'model_type':         'RandomForestClassifier',
            'target':             'MMP-9',
            'mcc_test':           mcc_champ,
            'split_strategy':     'Murcko scaffold',
            'warhead_smarts':     '[C](=O)[NH][OH]'
        }
        joblib.dump(metadata, os.path.join(model_dir, 'mmp9_rf_metadata.pkl'))

        print(f"\nSaved:")
        print(f"  mmp9_rf_champion.pkl         (MCC {mcc_champ:.4f}, threshold {self.optimal_threshold:.4f})")
        print(f"  mmp9_rf_explicit_warhead.pkl (2049 features)")
        print(f"  mmp9_rf_metadata.pkl")


if __name__ == "__main__":
    trainer = ModelTrainer()

    X_train, y_train, X_test, y_test, smiles_train, smiles_test = trainer.load_data(
        '../data/processed/mmp9_model_ready_splits.npz'
    )

    trainer.train(X_train, y_train)
    mcc_champ, y_pred_champ = trainer.evaluate(X_test, y_test)

    rf_hx = trainer.warhead_analysis(
        X_train, y_train, X_test, y_test,
        smiles_train, smiles_test, y_pred_champ
    )

    trainer.save_model(rf_hx, mcc_champ)
