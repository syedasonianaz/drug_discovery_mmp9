from huggingface_hub import hf_hub_download
import os

os.makedirs('models', exist_ok=True)

for f in ['mmp9_rf_champion.pkl', 'mmp9_rf_explicit_warhead.pkl', 'mmp9_rf_metadata.pkl']:
    hf_hub_download(
        repo_id='sonianaz/mmp9-inhibitor-predictor',
        filename=f'models/{f}',
        repo_type='space',
        local_dir='.'
    )
    print(f'Downloaded {f}')