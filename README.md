# fus-detectors

[![Paper PDF](https://img.shields.io/badge/paper-PDF-B31B1B.svg)](paper/preprint.pdf)
[![Extended Methods](https://img.shields.io/badge/methods-companion-4C6EF5.svg)](paper/methods_companion.pdf)
[![License: MIT](https://img.shields.io/badge/license-MIT-2EA44F.svg)](LICENSE)
[![Public CI](https://github.com/ArthurShune/fus-detectors/actions/workflows/public_ci.yml/badge.svg)](https://github.com/ArthurShune/fus-detectors/actions/workflows/public_ci.yml)

Post-clutter-suppression detection statistics that reduce artifact leakage in functional ultrasound and ultrafast Doppler maps, without changing the upstream clutter filter.

`fus-detectors` is the reference implementation and paper repository for localized matched-subspace detection on beamformed fUS slow-time data. The central question is simple: once a clutter-filtered residual has been fixed, can changing only the final detection statistic suppress artifacts more effectively than power-Doppler-style readouts?

## Why This Repo Matters

- It isolates the downstream detection statistic as a separate design choice on the same clutter-filtered residual.
- It keeps the existing acquisition and clutter-filtering pipeline intact.
- It provides a reusable same-residual evaluation protocol for synthetic, phantom, and real-IQ audits.

Two headline results anchor the repository:
- On the held-out `SIMUS-Struct-Intraop` benchmark, the fixed matched-subspace statistic reduces nuisance false-positive rate from `0.998` to `0.004` at matched recall `0.5` on the same clutter-filtered residual.
- On one open real-IQ rat-brain dataset, the fully whitened variant improves a conservative vessel-core versus perivascular-shell audit on all `10` evaluated blocks (`p = 0.002`).

> Status: active research code accompanying a preprint. APIs and helper scripts may still change. If you want to test this on raw fUS IQ, task data, or a clinical/mobile workflow, contact `arthur@skymesasystems.com`.

## Start Here

| If you want to... | Go here |
| --- | --- |
| Read the paper | [paper/preprint.pdf](paper/preprint.pdf) |
| See the full methods and companion analyses | [paper/methods_companion.pdf](paper/methods_companion.pdf) |
| Reproduce the headline figure and table | [scripts/reproduce_figure8_table7.sh](scripts/reproduce_figure8_table7.sh) |
| Prepare datasets | [docs/data_download.md](docs/data_download.md) |
| Cite the work | [CITATION.cff](CITATION.cff) |

## Paper

Arthur Shune, *Localized Matched-Subspace Detection for Functional Ultrasound and Ultrafast Doppler Imaging*, preprint, 2026.

[[PDF]](paper/preprint.pdf) · [[Extended Methods]](paper/methods_companion.pdf) · [[Supplement]](paper/supplement.pdf)

`arXiv link will be added once the preprint is posted.`

### Citation

```bibtex
@misc{shune2026localized,
  author = {Arthur Shune},
  title = {Localized Matched-Subspace Detection for Functional Ultrasound and Ultrafast Doppler Imaging},
  year = {2026},
  note = {Preprint},
  url = {https://github.com/ArthurShune/fus-detectors}
}
```

Machine-readable citation metadata is also provided in [`CITATION.cff`](CITATION.cff).

## Quick Start

```bash
conda env create -f environment.yml
conda activate fus-detectors
python scripts/verify_gpu.py
bash scripts/reproduce_figure8_table7.sh
```

After those commands, you will have the headline same-residual SIMUS figure and table used in the paper:
- `figs/paper/simus_detector_family_headline.pdf`
- `reports/simus_detector_family_ablation_table.tex`

Full paper reproduction requires the datasets listed below. Helper scripts do not auto-download them.

## Reproduce the Main Results

### Fast path

```bash
bash scripts/reproduce_figure8_table7.sh
```

Use this if you want the shortest paper-facing smoke test.

### Full manifest

For the command manifest used to build the reported paper artifacts, see:
- [`repro_manifest.json`](repro_manifest.json)
- [`paper/methods_companion.pdf`](paper/methods_companion.pdf)

## Datasets

Paper-scale reproduction uses generated synthetic data plus staged open datasets under `data/`.

| Dataset | Used for | Size | Expected location |
| --- | --- | --- | --- |
| `SIMUS/PyMUST` synthetic benchmark | Held-out structural benchmark | generated locally | `data/` not required |
| `ULM Zenodo 7883227` | Real-IQ vessel-core vs shell audit | about `50 GB` | `data/ulm_zenodo_7883227/` |
| `Shin RatBrain Fig3` | Real-IQ proxy-motion audit | about `6.5 GB` | `data/shin_zenodo_10711806/` |
| `Twinkling artifact / Gammex phantom` | Phantom structural audit | about `13.5 GB` extracted | `data/twinkling_artifact/` |
| `Whole-brain mouse fUS atlas` | Optional companion-only retrospectives | about `0.7 GB` | `data/whole-brain-fUS/` |

Detailed download links, expected filenames, and provider-specific caveats are in [`docs/data_download.md`](docs/data_download.md).

## Requirements

- Linux or WSL with `conda` or `mamba`
- NVIDIA GPU with CUDA recommended for paper-scale runs
- RTX 4080 SUPER (`16 GB` VRAM) is the reference GPU for the reported latency measurements

## For Collaborators

This repository is set up for same-residual comparisons on new data without changing the upstream clutter filter.

If you have raw complex IQ, beamformed slow-time data, or task-driven fUS acquisitions, a practical collaboration would look like this:
- you provide raw data, acquisition metadata, and any existing task or structural reference
- this code runs the fixed, adaptive, and fully whitened statistics on your existing clutter-filtered residual pipeline
- we return configured code, evaluation outputs, and a reusable harness you can run on later datasets

Contact: `arthur@skymesasystems.com`

## Repository Map

- `paper/` manuscript sources and built PDFs
- `pipeline/` detector statistics and clutter-filtered residual processing code
- `sim/` k-Wave and SIMUS simulation wrappers
- `scripts/` entry points for reported figures, tables, audits, and repro runs
- `reports/` generated LaTeX tables, CSV summaries, and audit artifacts
- `tests/` regression and unit tests

## License

This repository is released under the [MIT License](LICENSE).
