# Data acquisition checklist

Record every downloaded product before preprocessing.

## Required metadata

- Product and scene identifier
- Satellite/sensor and product level
- Acquisition date and time
- AOI and state
- Delivered bands and band order
- Pixel units and scale/offset
- CRS, pixel size and nodata value
- Cloud percentage, if supplied
- Source portal and license/usage constraints
- Local immutable raw path and checksum

## Initial AOIs

1. Assam floodplain
2. Meghalaya hilly forest
3. Tripura, Manipur or Nagaland agricultural region

Do not rename or modify source products in `data/raw`. Derived rasters belong under `data/processed`.
