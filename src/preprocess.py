import pandas as pd
import numpy as np
import os
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski

class MoleculeRefinery:
    def __init__(self, input_path: str):
        self.df = pd.read_csv(input_path)
        print(f"Loaded {len(self.df)} molecules.")

    def clean_and_convert(self):
        # 1. Drop rows missing critical data
        self.df = self.df.dropna(subset=['standard_value', 'canonical_smiles'])
        
        # 2. Fix the "Divide by Zero" - Remove values <= 0
        self.df = self.df[self.df['standard_value'] > 0]
        
        # 3. pIC50 Math
        self.df['standard_value'] = pd.to_numeric(self.df['standard_value']).clip(upper=100000000)
        self.df['pIC50'] = -np.log10(self.df['standard_value'] * (10**-9))
        return self

    def add_lipinski_descriptors(self):
        print("Validating molecules and calculating Lipinski...")
        
        # 1. Create RDKit Mol objects and keep track of which are valid
        mols = [Chem.MolFromSmiles(s) for s in self.df['canonical_smiles']]
        
        # 2. Add the mol objects to the dataframe temporarily to filter
        self.df['temp_mol'] = mols
        
        # 3. CRITICAL: Drop rows where RDKit returned None (Invalid SMILES)
        initial_count = len(self.df)
        self.df = self.df.dropna(subset=['temp_mol'])
        print(f"Removed {initial_count - len(self.df)} invalid molecules.")
        
        # 4. Now calculate descriptors safely
        self.df['MW'] = self.df['temp_mol'].apply(Descriptors.MolWt)
        self.df['LogP'] = self.df['temp_mol'].apply(Descriptors.MolLogP)
        self.df['NumHDonors'] = self.df['temp_mol'].apply(Lipinski.NumHDonors)
        self.df['NumHAcceptors'] = self.df['temp_mol'].apply(Lipinski.NumHAcceptors)
        
        # Drop the temporary mol column
        self.df = self.df.drop(columns=['temp_mol'])
        return self

    def save(self, output_filename: str):
        # Using the absolute path logic to ensure it saves in the right place
        current_file = os.path.abspath(__file__)
        base_path = os.path.dirname(os.path.dirname(current_file))
        processed_path = os.path.join(base_path, "data", "processed", output_filename)
        
        os.makedirs(os.path.dirname(processed_path), exist_ok=True)
        self.df.to_csv(processed_path, index=False)
        print(f"✅ Cleaned data saved to: {processed_path}")

if __name__ == "__main__":
    raw_path = r"E:\VS Code\machine_learning\drug_discovery\data\raw\mmp9.csv"
    refinery = MoleculeRefinery(raw_path)
    (refinery.clean_and_convert()
             .add_lipinski_descriptors()
             .save("mmp9_processed.csv"))