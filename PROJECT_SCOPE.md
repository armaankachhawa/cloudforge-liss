# CloudForge-LISS — Frozen Project Scope

## Project identity

**Name:** CloudForge-LISS  
**Subtitle:** Cloud-mask guided generative reconstruction for analysis-ready LISS-IV imagery

## Problem

Clouds and cloud shadows obscure surface information in LISS-IV optical imagery. CloudForge-LISS will automatically detect contaminated pixels and reconstruct only those pixels while retaining the original geospatial metadata and every valid clear-sky observation.

## MVP (Model 1)

**Input:** Cloudy LISS-IV bands + cloud mask + cloud-shadow mask  
**Model:** Small Attention ResUNet  
**Output:** Reconstructed LISS-IV + combined mask + confidence map

The initial supervised dataset will be made from clear LISS-IV targets corrupted with real cloud and shadow textures. GAN and diffusion models are explicitly out of scope until the deterministic baseline is trained and evaluated.

## Advanced mode (Model 2)

**Input:** Cloudy LISS-IV + masks + Sentinel-1 VV/VH + optional temporal LISS-IV  
**Model:** Multi-modal gated fusion network  
**Output:** Higher-confidence reconstructed LISS-IV

Advanced mode is optional at inference. The core LISS-only pipeline must remain operational without auxiliary data.

## Non-negotiable safeguards

1. Preserve clear pixels exactly; generated values are composed only into cloud/shadow pixels.
2. Operate on delivered multi-band raster values, never RGB screenshots.
3. Retain CRS, affine transform, dimensions, band order, nodata and relevant metadata.
4. Report uncertainty; never represent large occluded regions as guaranteed ground truth.
5. Split train/validation/test by scene, acquisition date and AOI—not random patches.
6. Calculate reference metrics inside the reconstruction mask.

## Target data coverage

- Assam floodplain: water, farmland, settlements and river edges.
- Meghalaya hills: forest, relief shadows and persistent clouds.
- Tripura, Manipur or Nagaland: agriculture, vegetation and small settlements.
- MVP target: 12–20 clear scenes, 8–12 cloudy scenes, at least three AOIs, 20–30 real mask sources, 3,000–8,000 patches and 3–5 independent test scenes.

## Required deliverables

- Reproducible preprocessing and scene manifest.
- Automatic three-class cloud-mask model: clear/cloud/shadow.
- No-AI masked baseline and Attention ResUNet reconstruction baseline.
- Optional Sentinel-1/temporal fusion experiment.
- Masked PSNR, SSIM, MAE, RMSE and SAM evaluation.
- NDVI and geospatial/edge validation where applicable.
- One-command GeoTIFF inference.
- Streamlit demonstration after model validation.

## Definition of done

`predict.py` accepts a georeferenced LISS-IV raster and creates:

- `reconstructed_liss.tif`
- `cloud_shadow_mask.tif`
- `confidence_map.tif`
- `preview_before_after.png`
- `quality_report.json`

The output raster must preserve source georeferencing and clear pixels exactly.
