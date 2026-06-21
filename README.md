# CloudForge-LISS

Cloud-mask guided reconstruction for analysis-ready LISS-IV satellite imagery.

The project is being implemented in dependency order: data audit, geospatial preprocessing, automatic masks, synthetic supervision, Attention ResUNet reconstruction, evaluation, confidence estimation and finally an operational demo.

The frozen objectives and scientific safeguards are in [PROJECT_SCOPE.md](PROJECT_SCOPE.md).

## Repository layout

```text
data/raw/                    Original LISS-IV and auxiliary products (not committed)
data/processed/aligned/      Co-registered rasters
data/processed/masks/        Cloud and shadow masks
data/processed/patches/      Training patches
data/processed/manifests/    Scene and patch manifests
src/                         Processing, training and inference code
configs/                     Reproducible experiment configuration
app/                         Streamlit demo (built after model validation)
checkpoints/                 Model weights (not committed)
outputs/                     Generated products (not committed)
tests/                       Automated tests
docs/                        Data cards and technical documentation
```

## Environment

Python 3.11 is the supported development version. Create the local environment with:

```powershell
uv venv --python 3.11 .venv
.\.venv\Scripts\Activate.ps1
uv pip install -r requirements.txt
```

PyTorch CUDA installation is machine-specific and will be selected only after verifying an NVIDIA GPU and driver. The code must also run on CPU for preprocessing and smoke tests.

## Current status

Project scope, repository skeleton and Python environment are ready. Dataset acquisition is paused at the Bhoonidhi account-login handoff; see [docs/PHASE_STATUS.md](docs/PHASE_STATUS.md).
