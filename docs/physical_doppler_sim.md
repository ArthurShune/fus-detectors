# Physical Doppler Simulation (Phase 0/1)

This repo's current Brain-* k-Wave pilots intentionally **do not** generate Doppler via moving scatterers or a moving medium. They:

1. run k-Wave once per steering angle,
2. beamform angle RF into complex images,
3. synthesize a slow-time cube, and then
4. apply **deterministic slow-time overlays** (flow Doppler band, alias tones, colored clutter, aperture phase screen) on the beamformed IQ cube.

Those overlays are excellent for controllable stress tests, but they are not a "physics-backed Doppler" forward model.

This document defines a minimal, low-regression-risk upgrade: a **moving-texture surrogate** that produces slow-time Doppler phase evolution from motion, while plugging into the existing acceptance-bundle pipeline.

## Goals

- **No detector changes**: do not modify STAP math or frozen evaluation profiles. Add a new simulation path only.
- **Canonical dataset artifact**: save a reproducible slow-time IQ cube `Icube` plus masks and provenance; derive acceptance bundles from that cube.
- **Doppler from motion**: slow-time phase evolution is induced by a motion model (advection) and Doppler phase increments, not by per-pixel injected sinusoids.
- **Natural aliasing**: aliasing should appear from PRF sampling when `|fD| > PRF/2`.
- **Telemetry-safe**: avoid periodic boundary/edge artifacts that would contaminate the repo's existing PSD/band-energy telemetry (peak-frequency occupancy, band ratios, etc.).

## Non-goals (Phase 1)

- Full channel-domain decorrelation or full-wave propagation per pulse.
- Field II backend integration (planned as a later backend tier).
- k-Wave-per-pulse backend (planned as an opt-in later backend tier).

## Contracts

### Canonical dataset artifact

The canonical output of the physical simulator is:

- `Icube`: complex64 array of shape `(T, Ny, Nx)` where:
  - `T` is slow-time frames sampled at `prf_hz`
  - `Ny` is depth (z) pixels
  - `Nx` is lateral (x) pixels
- `mask_flow`: boolean `(Ny, Nx)` (geometry-derived)
- `mask_bg`: boolean `(Ny, Nx)` (typically complement of flow, with optional guard bands)
- `mask_alias_expected` (optional): boolean `(Ny, Nx)` where expected `|fD| > prf_hz/2`

All lengths are uniform-grid and use repo-native geometry:

- `SimGeom.Nx`, `SimGeom.Ny`, `SimGeom.dx`, `SimGeom.dy` (meters)
- `SimGeom.f0` (Hz), `SimGeom.c0` (m/s)

### Derived acceptance bundle

Acceptance bundles are derived from the canonical dataset via:

- `sim/kwave/icube_bundle.py:write_acceptance_bundle_from_icube(...)`

This preserves compatibility with existing evaluation scripts and keeps detector math unchanged.

## Telemetry alignment (why some defaults matter)

This repo already has frozen Doppler-band conventions per regime and a telemetry stack that measures (among other things):

- band energies (e.g., `Eo`, `Ef`, `Eg`, `Ea`) and derived ratios like `malias = log((Ea+eps)/(Ef+eps))`
- a non-DC peak-frequency statistic `fpeak` used for band occupancy (e.g., "peak in Pf" fractions)

For Brain-* at `prf_hz=1500` and 64-frame windows, the default method conventions used in the repo are:

- `Pf = [30, 250] Hz`
- guard band `Pg = [250, 400] Hz`
- `Pa = [400, 750] Hz`

The physical simulator defaults below are chosen to avoid spurious periodicity (wrap-around), seams, or PSF choices that would show up as narrowband spectral lines or degenerate SVD/PSD telemetry.

## On-disk layout

The generator writes a run directory (user-specified `--out`) with:

```
<OUT>/
  dataset/
    icube.npy
    mask_flow.npy
    mask_bg.npy
    mask_alias_expected.npy        (optional)
    alias_observed_tile.npy        (optional; QC-only)
    config.json                    (frozen simulation config)
    meta.json                      (provenance + geometry + acquisition + file hashes)
    hashes.json                    (sha256 for dataset artifacts)
    debug/
      vx_mps.npy                   (optional)
      vz_mps.npy                   (optional)
      fd_expected_hz.npy           (optional)
      psf_calib.json               (optional; if using a calibrated PSF)
      alias_diag.json              (optional; QC-only thresholds/summary)
  bundle/
    <dataset_name>/                (standard acceptance bundle directory)
      meta.json
      mask_flow.npy / mask_bg.npy
      pd_base.npy / pd_stap.npy / ...
```

Notes:

- `dataset/` is the canonical audit artifact.
- `bundle/` is derived and may be regenerated from `dataset/` without rerunning the simulator.

## Phase 1 backend (moving-texture surrogate)

We represent tissue and blood as complex reflectivity textures on the `(Ny, Nx)` grid:

- tissue reflectivity is static (Phase 1 default)
- blood reflectivity is advected each PRF step by a velocity field inside a vessel mask

Let `dt = 1 / prf_hz`. For each time step:

1. **Advect** the blood reflectivity `R_blood` by semi-Lagrangian backtracing:
   - sample previous `R_blood` at `(x - vx*dt, z - vz*dt)`
   - use a padded working domain plus deterministic reservoir reinjection at inflow boundaries (avoid wrap-around periodicity)
2. **Apply Doppler phase increment** from axial motion (beam axis = z):
   - `delta_phi = (4*pi*f0/c0) * vz * dt`
   - `R_blood *= exp(1j*delta_phi)` inside the vessel
3. **Compose reflectivity**:
   - `R = R_tissue + a_blood * R_blood`
4. **Apply imaging operator** (Phase 1 default: parametric depth-dependent separable Gaussian PSF):
   - convolve real/imag parts with `sigma=(sigma_z_px(z), sigma_x_px(z))`
   - optionally, load a one-time calibration artifact (point target simulation) and fit `sigma_x(z), sigma_z(z)` for credibility
5. **Add complex AWGN** (optional) with an auditable SNR definition.

This is not full acoustics, but it is explicitly "no-tone": the slow-time phase evolution comes from motion and the Doppler relation, and aliasing arises naturally from PRF sampling.

### Deterministic reinjection default (avoid periodicity)

Default reinjection is designed so that:

- advection is the only source of temporal evolution (plus the Doppler phase term)
- no RNG calls are required inside the time loop (everything is "RNG once, then indexing")
- there is no wrap-around periodicity that could create spectral lines in the telemetry

Recommended default algorithm (Phase 1):

- Use a padded working domain `(Ny + 2*pad_z, Nx + 2*pad_x)`.
- Pre-generate a large complex "reservoir" texture `reservoir[Hres, Wres]` once from the run seed.
- On each slow-time step `t`, backtrace the padded coordinates via the velocity field and bilinear interpolation.
- Wherever the backtraced coordinate is out of bounds (and optionally wherever backtraces originate outside the vessel mask), reinject from the reservoir using a deterministic offset walk:
  - `ox = (ox0 + t * step_x) mod Wres`
  - `oz = (oz0 + t * step_z) mod Hres`
  - choose `step_x, step_z` co-prime with `(Wres, Hres)` (e.g., odd steps for power-of-two reservoir dims).

What to store in `meta.json` / `config.json`:

- padding: `pad_x`, `pad_z`
- reservoir shape: `Hres`, `Wres`
- reservoir seed and initial offsets: `reservoir_seed`, `ox0`, `oz0`
- offset steps: `step_x`, `step_z`
- interpolation mode (bilinear) and any boundary blending width (optional)

### PSF v1: parametric plus optional calibration artifact

Phase 1 baseline uses a real-valued, separable Gaussian PSF as an "imaging operator surrogate":

- lateral blur `sigma_x_px(z) = sigma_x0 + alpha_x * z_m`
- axial blur `sigma_z_px(z) = sigma_z0 + alpha_z * z_m`

where `z_m = z_index * dy` and all parameters are recorded in `config.json`.

For reviewer credibility without turning this into a calibration project, optionally support a point-target calibration artifact:

- run a one-time point target simulation (k-Wave or Field II) under the same acquisition geometry
- measure widths vs depth and fit `(sigma_x(z), sigma_z(z))`
- store the calibration file and its hash in `dataset/debug/psf_calib.json` and reference it from `meta.json`

### Alias labels and QC-only observed alias diagnostic

This simulator produces two conceptually distinct alias products:

- `mask_alias_expected`: pixel-level expected alias, defined from the velocity field:
  - `fd_expected_hz = (2*f0/c0) * vz`
  - `mask_alias_expected = mask_flow & (abs(fd_expected_hz) > prf_hz/2)`

- `alias_observed_tile` (optional): a QC-only tile-level diagnostic defined in the same language as the repo's existing band-energy telemetry:
  - compute per-tile band energies (`Ef`, `Ea`) and `malias = log((Ea+eps)/(Ef+eps))`
  - compute `fpeak` excluding DC
  - set a robust threshold using background tiles, e.g. `tau_alias = quantile_q(malias | Tbg)` with `q=0.99`
  - mark observed-alias tiles when `malias >= tau_alias` and (optionally) `fpeak` lies in `Pa`

Observed alias is **not** a ground-truth label and should never be used for performance claims. It exists to answer the question: "did alias-like energy emerge in the Pa band where expected?"

## Presets

Phase 1 provides two named presets (geometry + velocity-driven, not spectrum-injected):

- `microvascular_like`: Pf-dominant, broadened (Poiseuille profile; moderate `vmax`)
- `alias_stress`: contains regions with expected `|fD| > prf_hz/2`

Defaults are chosen to align with the Brain-* acquisition scale used throughout the repo:

- `Nx=Ny=240`, `dx=dy=90e-6 m`, `prf_hz=1500`, `T=5*64=320`, `f0=7.5e6 Hz`, `c0=1540 m/s`

## Reproducibility rules

- All randomness is derived from one `seed` and stored in `config.json`.
- For the default reinjection mode, avoid RNG calls inside the slow-time loop (RNG once, then deterministic indexing).
- `meta.json` includes:
  - exact CLI command
  - git commit + dirty flag
  - python + numpy + scipy versions
  - geometry/acquisition parameters
  - sha256 hashes for `icube.npy` and masks
- Record the simulator-specific "artifact safety" knobs:
  - reinjection padding + reservoir settings + offset walk parameters
  - PSF parameters and optional calibration artifact hash
  - alias diagnostic settings (if enabled) including band edges and quantiles
- `tests/` includes a fast smoke test asserting:
  - deterministic output for a fixed seed
  - nontrivial temporal variation within the flow region

## Sanity-Link Telemetry (Phase 2)

To support a defensible "physics-backed" story without in-vivo labels, we provide a
sanity-link script that computes summary slow-time statistics on:

- a generated physical-doppler `dataset/icube.npy` (with known flow/background masks), and
- real IQ cubes (Shin RatBrain Fig3 and/or Twinkling/Gammex RawBCF phantom cubes).

The script reports tile-level PSD band energies (Pf/Pg/Pa), non-DC peak frequency
occupancy, lag-1 coherence, and a small deterministic SVD summary on selected
tiles. Outputs are written under `reports/` and are deterministic given inputs.

Script:

- `scripts/physical_doppler_sanity_link.py`

Example (sim only):

```bash
PYTHONPATH=. conda run -n stap-fus \
  python scripts/physical_doppler_sanity_link.py \
  --sim-run runs/sim/phys_smoke_bundle \
  --out-dir reports/physdoppler_sanity_link \
  --tag sim_smoke
```

Example (sim + Shin + Gammex):

```bash
PYTHONPATH=. conda run -n stap-fus \
  python scripts/physical_doppler_sanity_link.py \
  --sim-run runs/sim/phys_smoke_bundle \
  --shin-root data/shin_zenodo_10711806/ratbrain_fig3_raw \
  --shin-iq-file IQData001.dat --shin-frames 0:128 --shin-prf-hz 1000 \
  --gammex-seq-root "data/twinkling_artifact/Flow in Gammex phantom" \
  --gammex-frames-along 0 --gammex-frames-across 0 --gammex-prf-hz 2500 \
  --out-dir reports/physdoppler_sanity_link \
  --tag sim_real_sanity
```

## Implementation notes

The initial Phase 1 implementation lives in:

- `sim/kwave/physical_doppler.py` (generator)
- `sim/kwave/pilot_physical_doppler.py` (CLI + dataset/bundle writer)

It currently implements:

- canonical dataset artifact (`Icube` + masks + provenance + hashes)
- deterministic padded-domain reinjection (reservoir + offset walk; no wrap-around by default)
- parametric depth-dependent Gaussian PSF (with optional JSON calibration override)
- expected-alias labeling (`mask_alias_expected`) plus a QC-only tile diagnostic (`alias_observed_tile.npy` + `alias_diag.json`)

Planned follow-ups:

- add a one-time point-target PSF calibration generator script (k-Wave or Field II) that writes `psf_calib.json` for reuse
- add a Field II backend (channel/RF-level moving scatterers) behind a backend interface, producing the same canonical `dataset/` layout
