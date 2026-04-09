import os
import pandas as pd

from ingest import DataIngestor
from preprocess import MoleculePreprocessor
from train import ModelTrainer
from screen import VirtualScreener

NPZ_PATH    = '../data/processed/mmp9_model_ready_splits.npz'
CLEAN_PATH  = '../data/processed/mmp9_clean.csv'
MODEL_DIR   = '../models/'
SCREEN_OUT  = '../data/processed/virtual_screening_hits.csv'


def main():
    print("Starting MMP-9 Drug Discovery Pipeline...")

    # --- 1. Ingestion ---
    ingestor = DataIngestor(target_id='CHEMBL321')
    ingestor.fetch_data()
    ingestor.clean_and_transform()
    ingestor.save_data(CLEAN_PATH)

    # --- 2. Preprocessing ---
    df_clean = pd.read_csv(CLEAN_PATH)

    preprocessor = MoleculePreprocessor()
    df_train, df_test = preprocessor.scaffold_split(df_clean)
    X_tr, y_tr, X_te, y_te, smi_tr, smi_te = preprocessor.prepare_matrices(df_train, df_test)
    preprocessor.save_processed_data(
        X_tr, y_tr, X_te, y_te, smi_tr, smi_te, NPZ_PATH
    )

    # --- 3. Training ---
    trainer = ModelTrainer()
    X_train, y_train, X_test, y_test, smiles_train, smiles_test = trainer.load_data(NPZ_PATH)
    trainer.train(X_train, y_train)
    mcc_champ, y_pred_champ = trainer.evaluate(X_test, y_test)

    rf_hx = trainer.warhead_analysis(
        X_train, y_train, X_test, y_test,
        smiles_train, smiles_test, y_pred_champ
    )
    trainer.save_model(rf_hx, mcc_champ, MODEL_DIR)

    # --- 4. Virtual Screening ---
    screener = VirtualScreener(
        os.path.join(MODEL_DIR, 'mmp9_rf_champion.pkl'),
        os.path.join(MODEL_DIR, 'mmp9_rf_metadata.pkl')
    )
    screener.fetch_chembl_approved()
    screener.run_screen(SCREEN_OUT)

    print("\nPipeline complete. Check data/processed/ for results.")


if __name__ == "__main__":
    main()
