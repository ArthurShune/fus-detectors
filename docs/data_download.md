# Dataset Download Notes (For Repro Runs)

This repo intentionally ignores large datasets under `data/` (see `.gitignore`). These notes document how to fetch the real-data inputs needed by the reproducibility + latency scripts.

## Shin RatBrain Fig3 (Zenodo 10711806)

Expected layout (used by `pipeline/realdata/shin_ratbrain.py` and `scripts/latency_realdata_rerun_check.py`):

- `data/shin_zenodo_10711806/ratbrain_fig3_raw/SizeInfo.dat`
- `data/shin_zenodo_10711806/ratbrain_fig3_raw/IQData001.dat` (and optionally `IQData002.dat`, ...)

Download + extract (Linux):

```bash
mkdir -p data/shin_zenodo_10711806
cd data/shin_zenodo_10711806

# ~6.0 GB
wget -c -O RatBrain_Fig3.zip \\
  https://zenodo.org/api/records/10711806/files/RatBrain_Fig3.zip/content

unzip RatBrain_Fig3.zip
mv RatBrain_Fig3 ratbrain_fig3_raw
```

Reference: DOI `10.5281/zenodo.10711806`.

## Twinkling Artifact / Gammex Phantom (MosMed, DOI 10.17816/DD76511)

The Gammex phantom scripts expect RawBCF cine files under:

- `data/twinkling_artifact/Flow in Gammex phantom/Flow in Gammex phantom (along - linear probe)/RawBCFCine.{par,dat}`
- `data/twinkling_artifact/Flow in Gammex phantom/Flow in Gammex phantom (across - linear probe)/RawBCFCine_08062017_145434_17.{par,dat}`

As of 2026-03-04, the MosMed dataset download flow is email-gated and uses a web form with captcha:

1. Open the MosMed dataset page:
   - `https://mosmed.ai/en/datasets/datasets/ultrasounddopplertwinklingartifact/`
2. Click `Download`.
3. Provide an email address and complete the captcha.
4. You will receive an email containing a download link and (if applicable) a password.
5. Download the dataset archive and extract it under `data/twinkling_artifact/`, preserving the folder names above.

Reference: DOI `10.17816/DD76511`.

