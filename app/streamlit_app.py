"""Minimal operational demonstration for validated CloudForge-LISS checkpoints."""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from src.predict import predict_scene

st.set_page_config(page_title="CloudForge-LISS", page_icon="☁️", layout="wide")
st.title("CloudForge-LISS")
st.caption("Mask-guided reconstruction for analysis-ready LISS-IV imagery")
st.warning("Generated pixels are estimates. Inspect the confidence layer before scientific use.")

uploaded = st.file_uploader("Cloudy LISS-IV GeoTIFF", type=["tif", "tiff"])
checkpoint_text = st.text_input("Validated checkpoint", "checkpoints/model1/best.pt")

if st.button("Run reconstruction", type="primary", disabled=uploaded is None):
    checkpoint = Path(checkpoint_text)
    if not checkpoint.exists():
        st.error(f"Checkpoint not found: {checkpoint}")
    else:
        with tempfile.TemporaryDirectory(prefix="cloudforge-") as temporary:
            temporary_path = Path(temporary)
            input_path = temporary_path / uploaded.name
            input_path.write_bytes(uploaded.getvalue())
            output_dir = temporary_path / "outputs"
            with st.spinner("Detecting contamination and reconstructing masked pixels…"):
                products = predict_scene(input_path, checkpoint, output_dir)
            st.success("Reconstruction completed")
            st.image(str(products["preview"]), caption="Cloudy (left) and reconstructed (right)")
            columns = st.columns(3)
            for column, key, label in zip(
                columns,
                ["reconstructed", "mask", "confidence"],
                ["Reconstructed GeoTIFF", "Cloud/shadow mask", "Confidence map"],
                strict=True,
            ):
                with column:
                    path = products[key]
                    st.download_button(label, path.read_bytes(), file_name=path.name)
