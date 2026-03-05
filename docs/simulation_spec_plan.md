# Simulation Spec: Repo-Realistic Phased Plan

This document maps `simulation_spec.txt` to what exists in this repository today and defines the next implementation phases in a low-regression-risk way. The core principle is: **do not touch detector math or frozen evaluation profiles**; add simulation backends and validation tooling only.

## Current Repo Reality (Pinned)

- Brain-* k-Wave pilots generate angle data and then apply **deterministic slow-time overlays on beamformed IQ**, i.e. Doppler/alias/clutter structure is imposed post-beamforming.
- Real-IQ motion stress tests inject motion at the **beamformed IQ** stage and explicitly do not model channel-domain decorrelation.

This is controllable and reproducible, but it is not a moving-scatterer forward model.

## Phase 0: Canonical Dataset Artifact (Done)

**Goal:** make `Icube(T,H,W)` the canonical artifact; derive acceptance bundles from it.

- Canonical dataset layout and bundle derivation:
  - `sim/kwave/icube_bundle.py:write_acceptance_bundle_from_icube(...)`

## Phase 1: Moving-Texture Physical Doppler Surrogate (Done)

**Goal:** generate slow-time Doppler from motion without per-pixel tone overlays; keep it fast and deterministic.

- Implementation:
  - `sim/kwave/physical_doppler.py`
  - `sim/kwave/pilot_physical_doppler.py`
- Design doc:
  - `docs/physical_doppler_sim.md`
- Key decisions already implemented:
  - deterministic padded-domain reservoir reinjection (no wrap-around by default)
  - parametric depth-dependent Gaussian PSF (optional JSON calibration override)
  - expected alias mask from velocity and QC-only observed-alias tile diagnostic

## Phase 2: “Sanity Link” Validation Against Real IQ (Done)

**Goal:** compute hard-to-game summary statistics (PSD band energies, peak occupancy, coherence, simple low-rank proxies) on simulation and on at least one open real IQ dataset.

- Implementation:
  - `scripts/physical_doppler_sanity_link.py`
- Outputs:
  - `reports/physdoppler_sanity_link/*_summary.json`
  - `reports/physdoppler_sanity_link/*_table.json`
- Recommended baseline runs:
  - Sim: any `runs/sim/*/dataset/icube.npy`
  - Shin Fig3: `data/shin_zenodo_10711806/ratbrain_fig3_raw/IQData001.dat` (frames `0:128`, PRF `1000`)
  - Gammex: `data/twinkling_artifact/Flow in Gammex phantom` (frame `0`, PRF `2500`)

## Phase 3: PSF Calibration Artifact (Done)

**Goal:** preempt “arbitrary PSF” critiques by offering an optional one-time calibration file used by Phase 1.

- Deliverable:
  - a script that writes `psf_calib.json` containing fitted `sigma_x0_px`, `sigma_z0_px`, `alpha_x_per_m`, `alpha_z_per_m`.
- Integration point (already supported):
  - `sim/kwave/physical_doppler.py:PsfSpec.calib_path`

Notes:
- This phase should not change defaults; calibration must be opt-in and hashed in dataset meta.

Status:
- Implemented calibration script:
  - `scripts/psf_calib_point_target_kwave.py`
- Dataset integration:
  - `sim/kwave/pilot_physical_doppler.py` embeds any provided `--psf-calib-path` as `dataset/debug/psf_calib.json` and hashes it.

## Phase 4: Field II Moving-Scatterer Backend (Plumbing Done, Exporter Pending)

**Goal:** add a “moving scatterers” backend tier without changing downstream evaluation.

- Plumbing implemented:
  - `scripts/fieldii_import_icube.py` imports an externally generated `Icube` (e.g. from MATLAB/Field II) into canonical `dataset/` + derived acceptance bundle.

Pending:
- A MATLAB/Field II exporter that writes:
  - `icube.npy` (complex, shape `(T,H,W)`)
  - optional `mask_flow.npy`, `mask_bg.npy`, `mask_alias_expected.npy`
  - (optional) a Field-II provenance JSON
- Once the exporter exists, the rest of the pipeline is already in place via the importer and bundle writer.

## Phase 5: k-Wave-per-Pulse Showcase (Deferred / Optional)

**Goal:** small-grid, small-T “full-wave” showcase regime for propagation realism, not the workhorse backend.

Notes:
- This is expensive and should be explicitly opt-in and kept separate from paper-critical throughput experiments.

## Open Decisions (Remaining)

- PSF calibration source and format:
  - k-Wave point target vs Field II point target vs empirical kernel table.
- Sanity-link acceptance criteria:
  - keep them qualitative/coarse (bucketed) and resistant to overfitting.
- Field II exchange contract:
  - exact coordinate conventions and metadata needed (dx/dz units, f0/c0, PRF, angle set if compounded).
