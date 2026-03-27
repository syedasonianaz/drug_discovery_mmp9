import sys
import os

# Import our custom logic from the other .py files
from ingest import DataIngestor
from preprocess import MoleculePreprocessor
from train import ModelTrainer
from screen import VirtualScreener

def main():
    print("🚀 Starting MMP-9 Drug Discovery Pipeline...")

    # --- 1. Ingestion ---
    # Downloads and cleans data from ChEMBL
    ingestor = DataIngestor(target_id='CHEMBL321')
    ingestor.fetch_data()
    ingestor.clean_and_transform()
    ingestor.save_data()

    # --- 2. Preprocessing ---
    # Handles Scaffold Splitting and Fingerprinting
    import pandas as pd
    df_clean = pd.read_csv('../data/processed/mmp9_clean.csv')
    
    preprocessor = MoleculePreprocessor()
    df_train, df_test = preprocessor.scaffold_split(df_clean)
    X_tr, y_tr, X_te, y_te = preprocessor.prepare_matrices(df_train, df_test)
    preprocessor.save_processed_data(X_tr, y_tr, X_te, y_te, '../data/processed/mmp9_model_ready.npz')

    # --- 3. Training ---
    # Trains the champion model and saves .pkl files
    trainer = ModelTrainer()
    trainer.train(X_tr, y_tr)
    trainer.evaluate(X_te, y_te)
    trainer.save_model()

    # --- 4. Screening ---
    # Final virtual screen for drug repurposing
    screener = VirtualScreener('../models/mmp9_rf_champion.pkl', '../models/mmp9_rf_metadata.pkl')
    screener.fetch_chembl_approved()
    hits = screener.run_screen()

    print("\n✅ Pipeline execution successful! Check the 'data/processed' folder for hits.")

if __name__ == "__main__":
    main()