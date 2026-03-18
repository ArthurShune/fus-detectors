# Dataset Setup Notes

Large datasets under `data/` are intentionally excluded from Git. These notes document the public inputs used by the reproduction and audit scripts, where to place them, and any download caveats.

## SIMUS / PyMUST synthetic benchmark

No external download is required. The held-out SIMUS structural benchmark is generated locally from the SIMUS/PyMUST toolchain.

## ULM Zenodo 7883227 (rat-brain kHz IQ)

Expected layout:

- `data/ulm_zenodo_7883227/param.mat`
- `data/ulm_zenodo_7883227/readme.pdf`
- `data/ulm_zenodo_7883227/IQ_001_to_025.zip`
- `data/ulm_zenodo_7883227/IQ_026_to_050.zip`
- `...`

The structural and motion scripts read directly from the `IQ_*.zip` archives and extract working files into `tmp/ulm_zenodo_7883227/` as needed.

Download:

1. Visit the Zenodo record: `https://doi.org/10.5281/zenodo.7883227`
2. Download the `IQ_*.zip` archives together with `param.mat` / `readme.pdf`
3. Place the files under `data/ulm_zenodo_7883227/`

Reference install footprint: about `50 GB` of zip archives.

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

Reference extracted footprint in the local test environment: about `13.5 GB`.

## Whole-brain mouse fUS atlas bundle (optional companion-only analyses)

This bundle is used only for companion-only PD retrospectives and atlas-aligned descriptive summaries.

Expected layout:

- `data/whole-brain-fUS/scan1.mat`
- `data/whole-brain-fUS/scan2.mat`
- `...`
- `data/whole-brain-fUS/allen_brain_atlas.mat`
- `data/whole-brain-fUS/scan_anatomy.mat`

Download:

1. Visit the Zenodo record: `https://doi.org/10.5281/zenodo.4905862`
2. Download the MATLAB bundle
3. Extract it under `data/whole-brain-fUS/`

Reference install footprint in the local test environment: about `0.7 GB`.
