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

## Pipeline commands

Audit downloaded rasters:

```powershell
.venv\Scripts\python.exe -m src.prepare_data audit
```

Align an auxiliary raster to a LISS-IV reference:

```powershell
.venv\Scripts\python.exe -m src.prepare_data align `
  --source data/raw/sentinel1/source.tif `
  --reference data/raw/liss_cloudy/reference.tif `
  --output data/processed/aligned/sentinel1.tif
```

Create a rough correction-ready mask and synthetic training patches:

```powershell
.venv\Scripts\python.exe -m src.create_masks --input scene.tif --output rough_mask.tif
.venv\Scripts\python.exe -m src.create_synthetic_clouds --count 3000 --patch-size 128
```

Train and evaluate Model 1:

```powershell
.venv\Scripts\python.exe -m src.train --config configs/model1.yaml
.venv\Scripts\python.exe -m src.evaluate `
  --prediction reconstructed.tif --target clear_target.tif --mask mask.tif
```

Operational inference:

```powershell
.venv\Scripts\python.exe -m src.predict `
  --input cloudy_liss.tif `
  --checkpoint checkpoints/model1/best.pt `
  --output outputs/run_001
```

Launch the final demo only after selecting a validated checkpoint:

```powershell
.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py
```
