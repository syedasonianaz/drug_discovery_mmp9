import os
import io
import joblib
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import gradio as gr
from rdkit import Chem
from rdkit.Chem import Draw

from predict import DrugPredictor

# ── Model loaded once at startup ─────────────────────────────────────────────
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_PATH, "models")

predictor = DrugPredictor(model_name="mmp9_rf_champion.pkl")
metadata  = joblib.load(os.path.join(MODEL_DIR, "mmp9_rf_metadata.pkl"))


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="MMP-9 Inhibitor Prediction API",
    description=(
        "QSAR binary classifier for MMP-9 inhibitory activity. "
        "Trained on ChEMBL321 with Murcko scaffold split. "
        "Hydroxamic acid warhead bias annotated explicitly on every prediction."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────────────
class SMILESRequest(BaseModel):
    smiles: str

    model_config = {
        "json_schema_extra": {
            "example": {"smiles": "O=C(NO)c1ccc(NC(=O)c2cccc(F)c2)cc1"}
        }
    }


class PredictionResponse(BaseModel):
    smiles:       str
    prediction:   str | None
    probability:  float | None
    warhead_flag: bool | None
    warhead_note: str | None
    error:        str | None


class HealthResponse(BaseModel):
    status:             str
    model_type:         str
    target:             str
    mcc_test:           float
    threshold:          float
    fingerprint_bits:   int
    fingerprint_radius: int
    split_strategy:     str


# ── API endpoints ─────────────────────────────────────────────────────────────
@app.get("/api/health", response_model=HealthResponse, tags=["API"])
def health():
    """Returns model metadata and deployment parameters."""
    return HealthResponse(
        status="ok",
        model_type=metadata["model_type"],
        target=metadata["target"],
        mcc_test=round(metadata["mcc_test"], 4),
        threshold=round(metadata["optimal_threshold"], 4),
        fingerprint_bits=metadata["fingerprint_bits"],
        fingerprint_radius=metadata["fingerprint_radius"],
        split_strategy=metadata["split_strategy"]
    )


@app.post("/api/predict", response_model=PredictionResponse, tags=["API"])
def predict(request: SMILESRequest):
    """
    Predicts MMP-9 inhibitory activity for a single SMILES string.
    Returns prediction, probability, and warhead flag.
    Warhead bias is always annotated explicitly — never silent.
    """
    result = predictor.predict(request.smiles)
    return PredictionResponse(smiles=request.smiles, **result)


@app.post("/api/batch", tags=["API"])
async def batch_predict(file: UploadFile = File(...)):
    """
    Accepts a CSV with a 'smiles' column.
    Optional columns: name, id, chembl_id, compound_name.
    Returns per-compound predictions and a summary.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="File must be a CSV.")

    contents = await file.read()
    try:
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
    except Exception:
        raise HTTPException(status_code=400, detail="Could not parse CSV file.")

    if "smiles" not in df.columns:
        raise HTTPException(
            status_code=400,
            detail="CSV must contain a 'smiles' column."
        )

    results = []
    for _, row in df.iterrows():
        smiles = str(row["smiles"])
        result = predictor.predict(smiles)
        entry  = {"smiles": smiles, **result}
        for col in ["name", "id", "chembl_id", "compound_name"]:
            if col in df.columns:
                entry[col] = row[col]
        results.append(entry)

    total   = len(results)
    active  = sum(1 for r in results if r.get("prediction") == "active")
    flagged = sum(1 for r in results if r.get("warhead_flag") is True)
    errors  = sum(1 for r in results if r.get("error") is not None)

    return {
        "summary": {
            "total":           total,
            "active":          active,
            "inactive":        total - active - errors,
            "warhead_flagged": flagged,
            "errors":          errors
        },
        "results": results
    }


# ── Gradio helper functions ───────────────────────────────────────────────────
def smiles_to_image(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Draw.MolToImage(mol, size=(400, 300))


def predict_single(smiles: str):
    if not smiles or not smiles.strip():
        return "Please enter a SMILES string.", None, None, None

    result = predictor.predict(smiles.strip())

    if result["error"]:
        return f"Error: {result['error']}", None, None, None

    prediction  = result["prediction"].upper()
    probability = result["probability"]
    warhead     = result["warhead_flag"]
    note        = result["warhead_note"]

    label = (
        "ACTIVE — Potential MMP-9 Inhibitor"
        if prediction == "ACTIVE"
        else "INACTIVE — Not predicted as MMP-9 Inhibitor"
    )
    prob_text    = f"Confidence: {probability:.1%}"
    warhead_text = (
        f"Hydroxamic Acid Warhead Detected\n\n{note}"
        if warhead
        else "No hydroxamic acid warhead detected."
    )

    return label, prob_text, warhead_text, smiles_to_image(smiles.strip())


def predict_batch(file):
    if file is None:
        return None, "Please upload a CSV file."

    try:
        df = pd.read_csv(file.name)
    except Exception as e:
        return None, f"Could not read CSV: {e}"

    if "smiles" not in df.columns:
        return None, "CSV must contain a 'smiles' column."

    results = []
    for _, row in df.iterrows():
        smiles = str(row["smiles"])
        result = predictor.predict(smiles)
        entry  = {
            "smiles":       smiles,
            "prediction":   result["prediction"],
            "probability":  result["probability"],
            "warhead_flag": result["warhead_flag"],
            "error":        result["error"]
        }
        for col in ["name", "id", "chembl_id", "compound_name"]:
            if col in df.columns:
                entry[col] = row[col]
        results.append(entry)

    results_df = pd.DataFrame(results)
    total   = len(results_df)
    active  = (results_df["prediction"] == "active").sum()
    flagged = results_df["warhead_flag"].sum()
    errors  = results_df["error"].notna().sum()

    summary = (
        f"Screening complete.\n"
        f"Total compounds: {total}\n"
        f"Active (predicted): {active}\n"
        f"Inactive: {total - active - errors}\n"
        f"Warhead flagged: {flagged}\n"
        f"Errors (invalid SMILES): {errors}"
    )

    return results_df, summary


# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(
    title="MMP-9 Inhibitor Predictor",
    theme=gr.themes.Base(
        primary_hue=gr.themes.Color(
    c50="#f0fdfa", c100="#ccfbf1", c200="#99f6e4",
    c300="#5eead4", c400="#2dd4bf", c500="#14b8a6",
    c600="#0d9488", c700="#0f766e", c800="#115e59",
    c900="#134e4a", c950="#042f2e"
),
        secondary_hue="slate",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("DM Mono"), "monospace"],
    ),
    css="""
        .container { max-width: 900px; margin: auto; }
        .header { text-align: center; padding: 2rem 0 1rem; }
        .header h1 { font-size: 2rem; letter-spacing: -0.02em; }
        .header p { color: #94a3b8; font-size: 0.95rem; }
        .warhead-box { border-left: 3px solid #f59e0b; padding-left: 1rem; }
        footer { visibility: hidden; }
    """
) as gradio_ui:

    with gr.Column(elem_classes="container"):

        gr.HTML("""
            <div class="header">
                <h1>MMP-9 Inhibitor Predictor</h1>
                <p>
                    QSAR binary classifier trained on ChEMBL321 · Random Forest · ECFP4 fingerprints<br>
                    Scaffold split · MCC 0.68 · Hydroxamic acid warhead bias annotated explicitly
                </p>
            </div>
        """)

        with gr.Tabs():

            with gr.Tab("Single Molecule"):
                with gr.Row():
                    with gr.Column(scale=2):
                        smiles_input    = gr.Textbox(label="SMILES", placeholder="Paste SMILES string here...", lines=2)
                        predict_btn     = gr.Button("Predict", variant="primary")
                        prediction_out  = gr.Textbox(label="Prediction", interactive=False)
                        probability_out = gr.Textbox(label="Confidence", interactive=False)
                        warhead_out     = gr.Textbox(label="Warhead Analysis", interactive=False, lines=4, elem_classes="warhead-box")
                    with gr.Column(scale=1):
                        mol_image = gr.Image(label="Structure", type="pil")

                gr.Examples(
                    examples=[
                        ["O=C(NO)c1ccc(NC(=O)c2cccc(F)c2)cc1"],
                        ["CC(=O)OC1=CC=CC=C1C(=O)O"],
                        ["CN1CCN(c2ccc(NC(=O)c3ccc(CN4CCN(C)CC4)cc3)cc2F)CC1"],
                    ],
                    inputs=smiles_input,
                    label="Example compounds"
                )

                predict_btn.click(
                    fn=predict_single,
                    inputs=smiles_input,
                    outputs=[prediction_out, probability_out, warhead_out, mol_image]
                )

            with gr.Tab("Batch Screening"):
                gr.Markdown("Upload a CSV with a **`smiles`** column. Optional columns: `name`, `id`, `chembl_id`.")
                file_input  = gr.File(label="Upload CSV", file_types=[".csv"])
                batch_btn   = gr.Button("Run Screening", variant="primary")
                summary_out = gr.Textbox(label="Summary", interactive=False, lines=7)
                results_out = gr.Dataframe(label="Results", interactive=False)

                batch_btn.click(
                    fn=predict_batch,
                    inputs=file_input,
                    outputs=[results_out, summary_out]
                )

            with gr.Tab("Model Info"):
                gr.Markdown("""
### About This Model

| Property | Value |
|---|---|
| Target | Matrix Metalloproteinase-9 (MMP-9) |
| ChEMBL ID | CHEMBL321 |
| Algorithm | Random Forest (n=300, max_depth=40) |
| Features | Morgan ECFP4 fingerprints (2048 bits) |
| Split strategy | Murcko scaffold split (80/20) |
| Class imbalance | class_weight='balanced' |
| Threshold selection | 5-fold CV on training set |
| Test MCC | 0.68 |

### Known Limitations

- **Hydroxamic acid warhead bias**: 64.5% of actives carry a hydroxamic acid warhead.
  Model predicts every warhead-carrying compound as active (TN=0 on that subset).
  Warhead presence is flagged explicitly on every prediction — never silent.

- **2D fingerprints only**: Cannot distinguish binding modes or 3D interactions.
  Top predictions validated with AutoDock Vina docking against 1GKC.

- **Scaffold generalization**: Train and test sets have mutually exclusive scaffolds.
  Performance on novel scaffold classes may vary.

### API Access

REST API available alongside this UI:

- `GET /api/health` — model metadata
- `POST /api/predict` — single SMILES prediction
- `POST /api/batch` — CSV bulk screening
- `GET /docs` — Swagger UI

### Intended Use

Exploratory virtual screening and hypothesis generation only.
Not a substitute for experimental validation.
                """)

        gr.HTML("""
            <div style="text-align:center; padding:1.5rem 0; color:#475569; font-size:0.8rem;">
                Built by Sonia · Biochemist & ML Practitioner · Karachi
                · <a href="https://github.com/syedasonianaz" style="color:#0d9488">GitHub</a>
            </div>
        """)


# ── Mount Gradio into FastAPI ─────────────────────────────────────────────────
app = gr.mount_gradio_app(app, gradio_ui, path="/")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
