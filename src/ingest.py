import pandas as pd
import numpy as np
from chembl_webresource_client.new_client import new_client
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
import os

class DataIngestor:
    """
    A class to handle the end-to-end pipeline of fetching, cleaning, 
    and preparing MMP-9 bioactivity data from ChEMBL.
    """
    
    def __init__(self, target_id='CHEMBL321'):
        self.target_id = target_id
        self.client = new_client
        self.raw_data = None
        self.clean_data = None

    def fetch_data(self):
        """Fetches raw IC50 bioactivity data for the target from ChEMBL."""
        print(f"Fetching data for target: {self.target_id}...")
        
        # Querying ChEMBL for human MMP-9 bioactivity entries
        activities = self.client.activity
        res = activities.filter(target_chembl_id=self.target_id) \
                        .filter(standard_type="IC50") \
                        .filter(standard_units="nM")
        
        self.raw_data = pd.DataFrame.from_dict(res)
        print(f"Downloaded {len(self.raw_data)} raw records.")
        return self.raw_data

    def clean_and_transform(self, ic50_threshold=1000):
        """
        Cleans data, removes duplicates, handles missing values, 
        and calculates pIC50 and bioactivity classes.
        """
        if self.raw_data is None:
            raise ValueError("No data found. Run fetch_data() first.")

        # 1. Drop missing values in essential columns
        df = self.raw_data.dropna(subset=['standard_value', 'canonical_smiles'])
        df['standard_value'] = df['standard_value'].astype(float)
        
        # 2. Convert IC50 (nM) to pIC50
        # Formula: -log10(IC50 * 10^-9)
        def convert_to_pic50(ic50_nm):
            molar = ic50_nm * 1e-9
            return -np.log10(molar)

        df['pIC50'] = df['standard_value'].apply(convert_to_pic50)
        
        # 3. Create Binary Labels (Active/Inactive)
        # 1.0 uM (1000 nM) is the standard threshold
        df['bioactivity_class'] = (df['standard_value'] <= ic50_threshold).astype(int)
        
        # 4. Remove duplicate SMILES (keeping the highest pIC50)
        df = df.sort_values('pIC50', ascending=False).drop_duplicates('canonical_smiles')
        
        self.clean_data = df[['molecule_chembl_id', 'canonical_smiles', 'pIC50', 'bioactivity_class']]
        print(f"Cleaning complete. {len(self.clean_data)} unique compounds remaining.")
        return self.clean_data

    def save_data(self, output_path='../data/processed/mmp9_clean.csv'):
        """Saves the final cleaned dataframe to a CSV file."""
        if self.clean_data is None:
            print("No cleaned data to save.")
            return

        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        self.clean_data.to_csv(output_path, index=False)
        print(f"Dataset saved to {output_path}")

# --- Execution Block ---
if __name__ == "__main__":
    # This block only runs if you execute 'python ingest.py' directly
    ingestor = DataIngestor()
    ingestor.fetch_data()
    ingestor.clean_and_transform()
    ingestor.save_data()