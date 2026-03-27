# MMP-9 Drug Discovery Pipeline
**AI-Driven Virtual Screening for Novel Cancer Therapeutics**

## 🔬 Project Overview
Matrix Metalloproteinase-9 (MMP-9) is a key enzyme involved in cancer metastasis. This project uses machine learning to identify potential inhibitors by training on known bioactivity data and screening FDA-approved drugs for repurposing opportunities.

## 📊 Performance Highlights
* **Model:** Optimized Random Forest (ECFP4 Fingerprints)
* **Champion MCC:** 0.6277 (Matthews Correlation Coefficient)
* **Custom Decision Threshold:** 0.4417 (Mathematically optimized for imbalanced chemical data)
* **Key Findings:** Successfully identified Zinc-binding ACE inhibitors (Enalapril) and potent cancer drugs (Ibrutinib) as high-probability MMP-9 hits.

## 🏗️ System Architecture (OOP)
The project is refactored from exploratory Jupyter Notebooks into a modular Python package:
* `ingest.py`: Automated ChEMBL API data retrieval and pIC50 transformation.
* `preprocess.py`: Structurally-aware Scaffold Splitting and SMOTE class balancing.
* `training.py`: Model serialization and metadata management.
* `screen.py`: Production-ready Virtual Screening with ADMET (Lipinski) filtering.

## 🚀 How to Run
1. Install dependencies: `pip install rdkit-pypi scikit-learn chembl_webresource_client pandas xgboost`
2. Run the pipeline:
   ```bash
   python src/ingest.py
   python src/preprocess.py
   python src/training.py
   python src/screen.py