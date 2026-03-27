import streamlit as st
import pandas as pd
import numpy as np
import joblib
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from PIL import Image

# --- Page Configuration ---
st.set_page_config(page_title="MMP-9 Drug Discovery AI", layout="wide")

# --- Load Model & Metadata ---
@st.cache_resource # Keeps the model in memory so it doesn't reload every click
def load_assets():
    model = joblib.load('../models/mmp9_rf_champion.pkl')
    metadata = joblib.load('../models/mmp9_rf_metadata.pkl')
    return model, metadata

model, meta = load_assets()
threshold = meta['optimal_threshold']

# --- Helper Functions ---
def get_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    return np.array(fp).reshape(1, -1)

# --- Sidebar ---
st.sidebar.title("🧬 MMP-9 Predictor")
st.sidebar.info(f"Model: {meta['model_type']}\n\nThreshold: {threshold:.4f}")

# --- Main UI ---
st.title("💊 AI-Powered Virtual Screening")
st.markdown("""
This application uses a **Random Forest** model trained on ChEMBL bioactivity data to identify potential inhibitors of **Matrix Metalloproteinase-9 (MMP-9)**.
""")

tab1, tab2, tab3 = st.tabs(["Single Molecule", "Batch Screening", "View Results"])

# --- Tab 1: Single Prediction ---
with tab1:
    st.header("Predict Single Compound")
    smiles_input = st.text_input("Enter SMILES string:", "C=CC(=O)N1CCC[C@@H](n2nc(-c3ccc(Oc4ccccc4)cc3)c3c(N)ncn32)C1")
    
    if st.button("Predict Bioactivity"):
        fp = get_fp(smiles_input)
        if fp is not None:
            prob = model.predict_proba(fp)[0, 1]
            is_active = prob >= threshold
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Probability", f"{prob:.2%}")
                if is_active:
                    st.success("Result: POTENTIAL INHIBITOR (Active)")
                else:
                    st.error("Result: INACTIVE")
            
            with col2:
                mol = Chem.MolFromSmiles(smiles_input)
                img = Draw.MolToImage(mol)
                st.image(img, caption="Chemical Structure")
        else:
            st.warning("Invalid SMILES string.")

# --- Tab 2: Batch Upload ---
with tab2:
    st.header("Batch Screening")
    
    some_upload = st.file_uploader("Upload CSV with 'smiles' column", type="csv")
    
    if some_upload:
        data = pd.read_csv(some_upload)
        
        if 'smiles' in data.columns:
            fps = [get_fp(s) for s in data['smiles']]
            valid_idx = [i for i, f in enumerate(fps) if f is not None]
            
            X = np.vstack([fps[i] for i in valid_idx])
            probs = model.predict_proba(X)[:, 1]
            
            results = data.iloc[valid_idx].copy()
            results['Probability'] = probs
            results['Prediction'] = ["Active" if p >= threshold else "Inactive" for p in probs]
            
            st.write(results.sort_values(by='Probability', ascending=False))
        else:
            st.error("CSV must contain a 'smiles' column.")

# --- Tab 3: Previous Screening Results ---
with tab3:
    st.header("Top Hits from ChEMBL Approved Drugs")
    try:
        hits = pd.read_csv('../data/processed/virtual_screening_hits.csv')
        st.dataframe(hits[['name', 'Active_Probability', 'chembl_id']].head(10))
    except:
        st.info("Run the virtual screening pipeline to see results here.")