# fus-detectors

Post-clutter-suppression detection statistics that reduce artifact leakage in functional ultrasound and ultrafast Doppler maps, without changing the upstream clutter filter.

`fus-detectors` is the reference implementation and manuscript repository for the preprint on localized matched-subspace detection for beamformed fUS slow-time data. The core question is deliberately narrow: once a clutter-filtered slow-time cube has been fixed, can changing only the final detection statistic reduce artifact leakage more effectively than power-Doppler-style readouts?

Two headline results anchor the repo:
- On the held-out SIMUS structural benchmark, the fixed matched-subspace statistic reduces nuisance false-positive rate from `0.998` to `0.004` at matched recall `0.5` on the same clutter-filtered residual.
- On one open real-IQ rat-brain dataset, the fully whitened variant shows a consistent structural improvement across all `10` audited blocks (`p = 0.002`) on a conservative vessel-core versus perivascular-shell endpoint.

> Status: active research code accompanying a preprint. Scripts and internal APIs may still change. If you want to test this on raw fUS IQ or task data, contact `arthur@skymesasystems.com`.

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

## Quick start

```bash
conda env create -f environment.yml
conda activate fus-detectors
python scripts/verify_gpu.py
bash scripts/reproduce_figure8_table7.sh
```

The last command regenerates the headline SIMUS figure and same-residual table used in the paper. Full benchmark reproduction requires the datasets listed below; helper scripts do not auto-download them.

## Reproduce the main results

The shortest paper-facing reproduction path is:

```bash
bash scripts/reproduce_figure8_table7.sh
```

This regenerates:
- `figs/paper/simus_detector_family_headline.pdf`
- `reports/simus_detector_family_ablation_table.tex`

For the full command manifest used to build the paper artifacts, see [`repro_manifest.json`](repro_manifest.json) and the companion document [`paper/methods_companion.pdf`](paper/methods_companion.pdf).

## Datasets

Paper-scale reproduction uses a mix of generated synthetic data and open public datasets staged under `data/`:

- `SIMUS/PyMUST synthetic benchmark`: generated locally, no external download required.
- `ULM Zenodo 7883227` (rat-brain kHz IQ): DOI `10.5281/zenodo.7883227`, about `50 GB` of zip archives in the reference setup. Place the downloaded `IQ_*.zip` files and metadata under `data/ulm_zenodo_7883227/`.
- `Shin RatBrain Fig3 (LOCA-ULM)` beamformed IQ: DOI `10.5281/zenodo.10711806`, about `6.5 GB`. Place the zip or extracted files under `data/shin_zenodo_10711806/`.
- `Twinkling artifact / Gammex phantom` RawBCF bundle: DOI `10.17816/DD76511`, about `13.5 GB` extracted in the reference setup. Place it under `data/twinkling_artifact/`. The provider currently uses an email-plus-captcha download flow.
- `Whole-brain mouse fUS atlas bundle` (optional companion-only PD retrospectives): DOI `10.5281/zenodo.4905862`, about `0.7 GB` in the reference setup. Place it under `data/whole-brain-fUS/`.

Detailed download notes, expected filenames, and provider-specific caveats are in [`docs/data_download.md`](docs/data_download.md).

## Requirements

- Linux or WSL with conda/mamba
- NVIDIA GPU with CUDA recommended for paper-scale runs
- The reported runtime numbers use an RTX 4080 SUPER (`16 GB` VRAM); that is the practical target for the heavier real-IQ and latency reproductions

## Repository map

- `paper/` manuscript sources and built PDFs
- `pipeline/` detector statistics and clutter-filtered residual processing code
- `sim/` k-Wave and SIMUS simulation wrappers plus ultrasound-specific runtime code
- `scripts/` entry points for reported figures, tables, audits, and repro runs
- `reports/` generated paper tables, CSV summaries, and audit artifacts
- `tests/` regression and unit tests for the method and data loaders

## License

This repository is released under the [MIT License](LICENSE).
