# fus-detectors

Code and manuscript artifacts for a preprint on localized matched-subspace detector heads for beamformed functional ultrasound (fUS) slow-time data after conventional clutter suppression.

The paper studies three downstream detector variants on the same residual cubes:
- a fixed matched-subspace detector, which is the default configuration
- an adaptive variant that selectively invokes whitening
- a fully whitened variant for regimes where additional covariance adaptation helps

Across the held-out SIMUS structural benchmark, the fixed variant is the strongest default downstream head on the same residual stream. In clutter-heavy stress tests, whitening helps in selected regimes. On one open real-IQ rat-brain dataset, the fully whitened variant shows a modest same-acquisition structural concentration advantage on a localization-derived vessel-core versus shell audit.

**Manuscript artifacts**
- Main preprint: [`paper/preprint.pdf`](paper/preprint.pdf)
- Extended methods and supplement companion: [`paper/methods_companion.pdf`](paper/methods_companion.pdf)
- Supplement-only build: [`paper/supplement.pdf`](paper/supplement.pdf)

## Quick start

```bash
conda env create -f environment.yml
conda activate fus-detectors
python scripts/verify_gpu.py
```

## One-command manuscript reproduction entry point

To regenerate the current headline same-residual SIMUS figure and table used in the main paper:

```bash
bash scripts/reproduce_figure8_table7.sh
```

This regenerates:
- Figure 8: `figs/paper/simus_detector_family_headline.pdf`
- Table 7: `reports/simus_detector_family_ablation_table.tex`

## Repository map

- `paper/` manuscript sources and built PDFs
- `pipeline/` detector and residualization code
- `sim/` simulation backends and ultrasound-specific runtime code
- `scripts/` experiment, figure, and table reproduction entry points
- `reports/` paper-facing generated tables and selected audit artifacts
- `tests/` focused regression and unit tests

Local-only private work and drafting material are intentionally excluded from this public repo. For untracked local clutter, use `.git/info/exclude`; a sample is provided at [`docs/git-info-exclude.sample`](docs/git-info-exclude.sample).
