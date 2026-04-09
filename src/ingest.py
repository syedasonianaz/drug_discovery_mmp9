import pandas as pd
import numpy as np
from chembl_webresource_client.new_client import new_client
import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


class DataIngestor:
    """
    Fetches, filters, and cleans MMP-9 bioactivity data from ChEMBL.
    Uses pChEMBL values directly (more reliable than manual IC50 conversion).
    Applies 3-class threshold logic, removes intermediates, outputs binary labels.
    """

    def __init__(self, target_id='CHEMBL321'):
        self.target_id = target_id
        self.client = new_client
        self.raw_data = None
        self.clean_data = None

    def fetch_data(self):
        """Fetches raw IC50 bioactivity data for the target from ChEMBL."""
        print(f"Fetching data for target: {self.target_id}...")
        res = self.client.activity
        res_query = res.filter(
            target_chembl_id=self.target_id
        ).filter(
            standard_type='IC50'
        )
        self.raw_data = pd.DataFrame.from_dict(res_query)
        print(f"Downloaded {len(self.raw_data)} raw records.")
        return self.raw_data

    def clean_and_transform(self):
        """
        Filters, classifies, and cleans bioactivity data.

        Filtering steps (matching notebook):
          1. Keep only exact measurements (standard_relation == '=')
          2. Drop rows missing pchembl_value or canonical_smiles
          3. Apply 3-class pChEMBL threshold:
               active:       pChEMBL >= 6.0  (IC50 <= 1 uM)
               inactive:     pChEMBL <= 5.0  (IC50 >= 10 uM)
               intermediate: 5.0 < pChEMBL < 6.0  -> removed
          4. Keep only binding assays (assay_type == 'B')
        """
        if self.raw_data is None:
            raise ValueError("No data found. Run fetch_data() first.")

        df = self.raw_data.copy()

        # 1. Exact measurements only
        before = len(df)
        df = df[df['standard_relation'] == '=']
        print(f"After relation filter: {before} → {len(df)} rows ({before - len(df)} removed)")

        # 2. Drop missing pChEMBL and SMILES
        before = len(df)
        df = df.dropna(subset=['pchembl_value', 'canonical_smiles'])
        df['pchembl_value'] = pd.to_numeric(df['pchembl_value'], errors='coerce')
        df = df.dropna(subset=['pchembl_value'])
        print(f"After null filter: {before} → {len(df)} rows ({before - len(df)} removed)")

        # 3. Three-class threshold — remove intermediates, keep binary
        print("\n--- Applying Activity Thresholds ---")
        before_class = len(df)

        def classify_activity(pchembl):
            if pchembl >= 6.0:
                return 'active'
            elif pchembl <= 5.0:
                return 'inactive'
            else:
                return 'intermediate'

        df['temp_class'] = df['pchembl_value'].apply(classify_activity)
        df = df[df['temp_class'] != 'intermediate'].copy()
        intermediates_dropped = before_class - len(df)

        df['bioactivity_class'] = df['temp_class'].map({'active': 1, 'inactive': 0})
        df = df.drop(columns=['temp_class'])

        print(f"Intermediate compounds removed: {intermediates_dropped}")
        print(f"After threshold filter: {len(df)} rows")
        print(df['bioactivity_class'].value_counts().to_string())

        # 4. Binding assays only
        before = len(df)
        df = df[df['assay_type'] == 'B']
        print(f"\nAfter assay type filter: {before} → {len(df)} rows ({before - len(df)} removed)")

        # Final columns
        self.clean_data = df[[
            'molecule_chembl_id',
            'canonical_smiles',
            'pchembl_value',
            'bioactivity_class'
        ]]

        total = len(self.clean_data)
        active_pct = self.clean_data['bioactivity_class'].sum() / total * 100
        inactive_pct = 100 - active_pct
        print(f"\nFinal shape: {self.clean_data.shape}")
        print(f"Active: {self.clean_data['bioactivity_class'].sum()} ({active_pct:.2f}%)")
        print(f"Inactive: {(self.clean_data['bioactivity_class']==0).sum()} ({inactive_pct:.2f}%)")
        print("Data is highly imbalanced — class_weight='balanced' handles this in training.")

        return self.clean_data

    def save_data(self, output_path='../data/processed/mmp9_clean.csv'):
        """Saves the cleaned dataframe to CSV."""
        if self.clean_data is None:
            print("No cleaned data to save.")
            return
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.clean_data.to_csv(output_path, index=False)
        print(f"Dataset saved to {output_path}")


if __name__ == "__main__":
    ingestor = DataIngestor()
    ingestor.fetch_data()
    ingestor.clean_and_transform()
    ingestor.save_data()
