# Draft Provisional Specification

This is a first-pass technical specification draft for a U.S. provisional application. It is not legal advice and is not a substitute for attorney review. The purpose of this draft is to capture the invention with enough breadth and technical detail to support a later nonprovisional filing without new-matter problems.

This draft is intentionally organized around invention families rather than around the manuscript or benchmark narrative. The results in the paper are examples of utility; they are not the invention itself.

## Title

Systems and Methods for Localized Matched-Subspace Detection in Ultrasound Doppler Imaging

Alternative titles:

- Systems and Methods for Localized Post-Residual Flow Detection in Beamformed Ultrasound Data
- Systems and Methods for Adaptive Local Whitening and Matched-Subspace Scoring in Ultrafast Doppler Imaging
- Systems and Methods for Localized Flow-Consistent Signal Detection in Beamformed Ultrasound Data

## Inventor / Applicant Placeholders

- Inventor(s): `[confirm full legal names]`
- Assignee / Applicant: `[confirm entity, e.g. SkyMesa Systems Inc.]`
- Correspondence: `[counsel or applicant address]`
- Government-interest statement: `[confirm whether any is required]`

## Field

The disclosure relates generally to ultrasound signal processing, including functional ultrasound, ultrafast Doppler imaging, power Doppler imaging, color flow imaging, microvascular flow detection, and post-residual or post-clutter-suppression detection in beamformed complex slow-time data.

## Definitions and Construction Notes

Unless the context requires otherwise, the following definitions may be used in the specification and later claim drafting:

- `beamformed IQ` means complex-valued ultrasound data after beamforming and before or after one or more downstream filtering stages
- `residual data` or `residual cube` means beamformed slow-time data after a clutter-suppression, residualization, or clutter-reduction stage
- `tile` means a spatially localized support region, which may overlap adjacent support regions
- `slow-time snapshot` means a temporal vector or embedded temporal sample derived from one or more localized support regions
- `flow band` means a target temporal-frequency region or equivalent target subspace associated with expected flow-consistent signal
- `guard band` means a temporal-frequency region or equivalent subspace outside the target flow band used to measure nuisance or clutter evidence
- `alias band` means a temporal-frequency region or equivalent subspace associated with aliased or higher-velocity nuisance energy
- `matched-subspace score` means a score based on energy or projected energy aligned with a target subspace relative to one or more non-target components
- `fixed detector` means a non-whitened or identity-whitened matched-subspace detector branch
- `fully whitened detector` means a covariance-adaptive detector branch using localized covariance conditioning and whitening
- `adaptive detector` means a branch-selection or blended detector that chooses between fixed and whitened processing using clutter evidence
- `shrink-only penalty` means a post-score transform that can decrease but not increase scores
- `candidate set` means a region eligible for shrink-only suppression
- `protected set` means a region excluded from shrink-only suppression
- `score map` means any map, image, or per-location output produced from localized detector scores

These definitions are illustrative and should be broadened rather than narrowed during final attorney review.

## Background

Ultrafast Doppler and functional ultrasound systems commonly perform a clutter-suppression stage followed by a downstream flow readout such as power Doppler, log-power Doppler, autocorrelation-based Doppler magnitude, or related energy statistics. In practice, a clutter-suppression stage may remove dominant tissue or stationary clutter while leaving structured nuisance, phase instability, aliased flow, specular leakage, vibration, probe motion, or local nonstationary interference in the residual signal. Conventional power-based downstream readouts can rank such nuisance highly, particularly in operating regimes where low false-positive behavior is important.

This is especially problematic in use cases where a small number of false detections can distort vascular interpretation, functional maps, thresholded displays, or downstream statistics. Examples include intra-operative imaging, cranial-window imaging, bedside imaging, mobile imaging, and other settings in which residual motion, clutter leakage, or artifact-heavy scenes can cause rare but high-scoring false positives.

Conventional approaches often focus on adapting the clutter filter itself. However, even after a reasonable residualization stage, the downstream detector remains an independent design choice. What is needed is a downstream detector framework that can operate on beamformed ultrasound slow-time residuals, locally distinguish flow-consistent signal from structured nuisance, and do so with embodiments ranging from simple fixed scoring to more advanced adaptive whitening, while optionally providing a bounded, safety-oriented score suppression layer and deployment-oriented threshold transfer workflows.

## Summary

The disclosure provides systems, methods, and non-transitory computer-readable media for localized post-residual flow detection in beamformed ultrasound slow-time data.

In one aspect, a residual slow-time signal is obtained from beamformed ultrasound data after a clutter-suppression stage. The residual is processed tilewise or regionally rather than only globally. For each tile or region, the method forms one or more slow-time vectors or embedded slow-time snapshots and computes a score that measures energy aligned with a designated flow subspace relative to non-flow or out-of-band energy. In some embodiments, the score is a matched-subspace or subspace-energy score. In some embodiments, the score is self-normalized. In some embodiments, the score is aggregated over overlapping tiles to produce a per-pixel or per-location output map.

In another aspect, the disclosure provides multiple detector branches. A fixed branch computes a non-whitened or identity-whitened matched-subspace score. A fully whitened branch computes a covariance-adaptive score based on a locally estimated covariance matrix. An adaptive branch selectively chooses between the fixed and whitened branches based on one or more clutter-evidence features, such as guard-band energy, alias-band energy, nuisance-risk statistics, motion indicators, covariance-condition indicators, or combinations thereof.

In another aspect, the disclosure provides an optional shrink-only penalty layer applied to detector scores. In some embodiments, the penalty layer computes a multiplicative weight or other monotone non-increasing transform from side-information features. In some embodiments, the penalty layer can reduce scores but cannot increase them. In some embodiments, the penalty layer preserves a protected set of locations, such as flow-proxy or vessel-proxy locations, and defaults to an inert mode when side-information is non-actionable, missing, flat, or inconsistent.

In another aspect, the disclosure provides threshold-calibration and transfer workflows. Thresholds may be learned from separate negative-only calibration data, nuisance banks, separate blocks, separate sessions, separate subjects, or separate devices, and then transferred to evaluation data without per-window retuning. In some embodiments, the threshold targets a right-tail operating point, a fixed false-positive rate, or a matched-sensitivity operating point.

In another aspect, the disclosure provides system and deployment embodiments, including scanner-integrated embodiments, workstation embodiments, GPU or accelerator embodiments, edge embodiments, and non-transitory computer-readable-medium embodiments. In some embodiments, the disclosure further provides real-time or near-real-time execution architectures, including tiled batching, temporal embedding, local covariance estimation, conditional execution of whitening only where needed, overlap-add stitching, cached geometry, graph-replay execution, and other exact-output-preserving optimizations.

The disclosed technology therefore provides a detector platform that can improve nuisance discrimination and operating-point stability in ultrasound Doppler imaging without requiring replacement of the upstream clutter filter.

## Brief Description of the Drawings

The following drawing list is recommended for the provisional filing. The provisional does not need publication-quality figures, but the drawings should be clear and technically complete.

1. **Figure 1: End-to-end processing overview.**
   A block diagram showing beamformed IQ input, residualization, tile extraction, fixed detector, adaptive detector, fully whitened detector, optional shrink-only penalty, and output map.

2. **Figure 2: Tile extraction and temporal embedding.**
   A schematic showing extraction of overlapping spatial tiles from a beamformed slow-time cube, formation of tile-local slow-time vectors or sliding-window embeddings, and aggregation of local results.

3. **Figure 3: Doppler band geometry.**
   A schematic or spectrum plot showing clutter band, flow band, guard band, and alias band and the corresponding band operators or filters.

4. **Figure 4: Fixed detector branch.**
   A flow diagram showing localized score generation without local whitening.

5. **Figure 5: Fully whitened detector branch.**
   A flow diagram showing localized covariance estimation, conditioning, whitening, matched-subspace scoring, and map aggregation.

6. **Figure 6: Adaptive branch selection.**
   A diagram showing a clutter-evidence feature feeding a switch or gate between a fixed branch and a whitened branch.

7. **Figure 7: Shrink-only penalty layer.**
   A diagram showing candidate-set determination, protected-set determination, shrink-only weight generation, inert-mode logic, and post-penalty output.

8. **Figure 8: Threshold-transfer workflow.**
   A diagram showing calibration-bank threshold learning and application to held-out data without per-window recalibration.

9. **Figure 9: Real-time processing architecture.**
   A system diagram showing tile batching, hardware acceleration, optional conditional execution, overlap-add stitching, and output generation.

10. **Figure 10: System implementation embodiments.**
   A diagram showing a scanner-integrated embodiment, workstation embodiment, and edge-compute embodiment.

11. **Figure 11: Medium embodiment.**
   A diagram showing instructions stored on one or more machine-readable media and executed by one or more processors to perform the disclosed method.

12. **Figure 12: Example comparative output maps.**
   Example output maps showing a baseline power-based map and one or more disclosed detector outputs on the same residual input.

Recommended source material in the repo:

- [figs/paper/stap_pipeline_bayes_block.pdf](/home/arthu/stap-for-fus/figs/paper/stap_pipeline_bayes_block.pdf)
- [figs/paper/hankelization_local_stap.pdf](/home/arthu/stap-for-fus/figs/paper/hankelization_local_stap.pdf)
- [figs/paper/doppler_band_geometry_psd.pdf](/home/arthu/stap-for-fus/figs/paper/doppler_band_geometry_psd.pdf)
- [figs/paper/shrink_only_tail_suppression.pdf](/home/arthu/stap-for-fus/figs/paper/shrink_only_tail_suppression.pdf)

## Detailed Description

### 1. Overview

In an exemplary embodiment, a system receives beamformed complex ultrasound data having a spatial dimension and a slow-time dimension. The system applies or receives the output of a clutter-suppression or residualization stage, then applies a localized detector head to the residual signal. The detector head may be a fixed non-whitened branch, a fully whitened branch, or an adaptive branch that selectively applies whitening. In some embodiments, an optional shrink-only penalty is then applied. The resulting score map may be used for detection, display, thresholding, mapping, monitoring, vascular imaging, functional imaging support, or other downstream processing.

The disclosed techniques are not limited to any particular upstream residualizer. The residualizer can be any transform, filter, or model that produces a residual slow-time signal, including singular-value decomposition, adaptive singular-value decomposition, blockwise local singular-value decomposition, robust PCA, HOSVD, learned clutter suppression, temporal filtering, or combinations thereof.

The disclosure is also not limited to any one dataset, benchmark, frame count, tile size, pulse repetition frequency, covariance recipe, hardware platform, or evaluation protocol. Named datasets and experiments in the associated manuscript should be treated as non-limiting examples of utility.

### 2. Input Data and Residualization

The input may comprise beamformed in-phase and quadrature ultrasound data or any equivalent complex slow-time ultrasound signal arranged over time and space. In some embodiments, the input is a complex-valued slow-time cube `X[t, y, x]`. In other embodiments, the data are arranged in other tensor layouts. The system may operate on a full frame, a block, a sliding window, or a selected region of interest.

A residualization stage may be applied before detector scoring. In some embodiments, this stage is motion-compensated singular-value decomposition. In some embodiments, it is a low-rank clutter filter. In some embodiments, it is a learned or hybrid clutter suppressor. In some embodiments, the detector is applied after beamforming and after residualization. In some embodiments, the residualization stage is treated as separate from the detector head so that different downstream score heads can operate on the same residual.

### 3. Spatial Localization and Temporal Support Formation

The residual data are processed in spatially localized fashion. In one embodiment, the image plane is partitioned into overlapping tiles. In another embodiment, regions, patches, superpixels, adaptive windows, or multiscale neighborhoods are used.

For each localized support region, the system forms one or more slow-time vectors. The vectors may be formed directly from temporal samples at a spatial location, pooled across a tile, or generated by sliding-window temporal embedding. In some embodiments, a Hankel or similar embedding is used to generate multiple overlapping temporal snapshots from each location. In some embodiments, the number of snapshots is capped or subsampled. In some embodiments, the training support is pooled over multiple pixels, frames, windows, or combinations thereof.

### 4. Doppler Band Geometry

In some embodiments, the detector uses explicit Doppler or temporal-frequency partitions. These may include one or more of:

- a clutter or direct-current band
- a target flow band
- a guard band adjacent to or near the target flow band
- an alias or high-velocity nuisance band
- one or more out-of-band complements

The bands may be implemented as exact orthogonal projectors, approximate projectors, filters, analysis operators, Fourier-domain selection operators, wavelet-domain operators, or other subspace representations. Band definitions may depend on pulse repetition frequency, acquisition settings, the number of slow-time samples, or other metadata.

In some embodiments, the bands are fixed once per operating regime. In some embodiments, the bands are adapted. In some embodiments, empty or degenerate bands are detected by deterministic sentinels, and a branch or penalty layer is disabled when discretization is invalid.

### 5. Fixed Matched-Subspace Detector

In one embodiment, the system computes a non-whitened or identity-whitened band-limited matched-subspace score. For a localized slow-time vector `x`, a detector score may depend on the energy of `x` in the target flow subspace relative to total energy, out-of-band energy, orthogonal-complement energy, or a normalized denominator.

An example score family is:

`S_fixed = flow_band_energy / (complement_energy + rho * total_energy + epsilon)`

where `rho` and `epsilon` are stabilization terms and the flow-band energy is computed using a band or subspace operator associated with target flow.

The exact formula can vary. The important point is that the fixed branch uses localized band-limited matched-subspace scoring without local covariance whitening. This branch can be deployed everywhere by default, especially where simplicity, stability, or lower compute cost is preferred.

### 6. Fully Whitened Detector

In another embodiment, the system computes a covariance-adaptive fully whitened detector score.

For a given localized support, the system estimates a local covariance matrix from one or more training snapshots. The covariance matrix may be estimated using:

- a sample covariance matrix
- a shrinkage covariance
- a robust M-estimator
- a Tyler estimator
- a Huber estimator
- a trimmed or outlier-robust covariance estimator
- a diagonally loaded covariance
- combinations thereof

The covariance is then conditioned if desired, for example by shrinkage, diagonal loading, trace normalization, eigenvalue flooring, condition-number targeting, or combinations thereof. A whitening transform is derived from the conditioned covariance, and the localized data are whitened before scoring.

In some embodiments, the fully whitened score is a matched-subspace ratio score. In some embodiments, it is a whitened power score. In some embodiments, it is a GLRT-style statistic or other covariance-adaptive energy detector. In some embodiments, the detector uses the same target flow band geometry as the fixed branch but differs in its covariance-adaptive pre-processing.

### 7. Adaptive Detector

In another embodiment, the system adaptively selects between a fixed branch and a whitened branch.

In one implementation, a clutter-evidence feature is computed for each tile, pixel, or region. The feature may be derived from:

- guard-band energy
- guard-band fraction
- alias-to-flow ratio
- alias-band energy
- motion magnitude
- a registration residual
- a covariance condition metric
- support quality metrics
- combinations thereof

The system compares the clutter-evidence feature to one or more thresholds, rules, or models. If clutter evidence is weak, the system uses the fixed detector branch. If clutter evidence is strong, the system uses the whitened detector branch. In some embodiments, the switching is binary. In some embodiments, the branches are blended. In some embodiments, the switching is probabilistic or multi-level.

One example embodiment uses a baseline guard-band energy fraction `r_g` and a threshold `tau_g`, such that the whitened branch is enabled when `r_g >= tau_g`. However, the invention is not limited to that exact feature or threshold rule.

In other embodiments, the switching logic may be implemented as a soft weighting between branches, a confidence-weighted combination, a multilevel selection among more than two branches, or a frame-level or region-level policy determined from one or more clutter-evidence features.

### 8. Score Aggregation

Localized scores may be aggregated to form an output map. In some embodiments, all tiles overlapping a pixel contribute to that pixel’s final score. In some embodiments, overlap-add, weighted averaging, confidence weighting, interpolation, or voting is used. In some embodiments, different branches may be aggregated separately and then combined.

The output may be a per-pixel score map, a per-region score map, a thresholded detection map, a vascular map, a heat map, a risk map, or another representation suitable for downstream interpretation or inference.

### 9. Shrink-Only Penalty Layer

In another embodiment, the system applies an optional shrink-only penalty layer to detector scores.

The penalty layer may use side-information features derived from the same local data or from auxiliary maps. Example features include:

- flow-band energy
- guard-band energy
- alias-band energy
- alias-to-flow ratios
- vessel-proxy support
- flow-proxy coverage
- local background-like indicators
- spatial support consistency
- motion or clutter risk indicators

The penalty layer may define:

- a candidate set eligible for score reduction
- a protected set that cannot be penalized
- an activation condition
- an inert mode

In some embodiments, the penalty is multiplicative and uses a weight `w` with `0 < w <= 1`, such that `post_score = w * pre_score`. In some embodiments, the transform is any monotone non-increasing transform. In some embodiments, the penalty never increases any score. In some embodiments, the system records explicit reasons when the penalty layer is disabled.

The penalty layer may be especially useful for artifact-heavy scenes, for reducing extreme right-tail nuisance, or for preserving a safety-oriented deployment behavior.

### 10. Integrity Controls and Safe Failure Modes

In some embodiments, the system includes deterministic safeguards to avoid invalid processing or misleading results. These may include:

- detection of invalid band discretization
- detection of missing or inconsistent metadata
- suppression of penalty activation when features are flat or non-actionable
- disabling of adaptive branches when support quality is insufficient
- proxy-mask circularity controls
- leak-prevention controls when conditional execution is used
- explicit logging of failure modes or fallback states

In some embodiments, a fallback to the fixed detector is a valid success mode rather than a failure.

### 11. Threshold Calibration and Transfer

The detector outputs may be thresholded using empirical calibration.

In some embodiments, thresholds are learned from negative-only calibration banks, nuisance banks, separate calibration sessions, held-out calibration windows, separate subjects, or separate devices. In some embodiments, thresholds target a selected right-tail false-positive rate. In some embodiments, thresholds are transferred to evaluation data without per-window recalibration. In some embodiments, the transfer is across windows, blocks, animals, scans, sessions, probes, or scanners.

In some embodiments, different detector branches use different fixed thresholds. In some embodiments, a common threshold family is used. In some embodiments, threshold transfer is combined with branch selection or adaptive gating.

The disclosure is not limited to any one calibration source. Calibration may be performed using reference scans, nuisance banks, background banks, separate subjects, separate animals, separate sessions, scanner-specific calibration data, or other designated calibration corpora.

### 12. Real-Time and Hardware Acceleration Embodiments

In some embodiments, the detector is executed in real time or near real time.

Example implementation features include:

- batched tile extraction
- tensor-unfold or equivalent tile generation
- batched temporal embedding
- batched local covariance estimation
- batched whitening and band-energy scoring
- exact-output-preserving cached overlap counts or geometry terms
- conditional execution that avoids whitening in tiles where it is not selected
- overlap-add reconstruction
- graph capture or replay to reduce launch overhead
- GPU, FPGA, ASIC, DSP, CPU, scanner backend, edge device, or cloud execution

In some embodiments, diagnostics not required for output generation are omitted in latency-sensitive modes while preserving the exact output scores.

### 13. System and Medium Embodiments

In some embodiments, the disclosed detector is implemented in an ultrasound scanner, an ultrasound research platform, a workstation connected to an ultrasound scanner, a portable or edge-computing device, a remote compute server, or a cloud-connected processing system.

In some embodiments, a system includes:

- an ultrasound data interface configured to receive beamformed IQ data or residualized slow-time data
- one or more processors
- memory storing instructions
- one or more displays or output interfaces

The processors may be configured to perform any combination of the fixed detector, adaptive detector, fully whitened detector, shrink-only penalty layer, threshold transfer, aggregation, and output-generation steps described herein.

In some embodiments, a non-transitory computer-readable medium stores instructions that, when executed by one or more processors, cause the processors to perform any of the methods described in this specification.

### 14. Exemplary Operating Regimes

The invention is not limited to any one operating setting, but some embodiments may use:

- tile sizes such as `8x8` or other sizes
- overlapping tiles with selected strides
- slow-time support lengths such as `L_t = 8`, `16`, `32`, `64`, or others
- fixed or adaptive Doppler bands
- covariance loading or shrinkage
- robust covariance recipes
- a fixed detector as a default branch
- an adaptive branch for clutter-heavy settings
- a fully whitened branch for specialist settings

These numerical values are examples only and should not be treated as limiting.

### 15. Alternative Embodiments

The following alternatives should be understood as within the scope of the disclosed system:

- applying the detector to pre-beamformed, partially beamformed, or fully beamformed data
- using other subspace constructions besides Fourier bands
- using data-driven or learned nuisance features to trigger whitening
- using more than two detector branches
- combining multiple branch outputs
- using the disclosed detector as an intermediate stage before downstream functional or vascular analysis
- applying the shrink-only layer before or after spatial aggregation
- using different protected-set logic
- using one or more reference maps, anatomical priors, or spatial context features as side information
- using the detector in vascular, functional, structural, interventional, diagnostic, or monitoring applications

### 16. Best-Known Current Implementations

The provisional should disclose the best-known current implementations available at the time of filing while preserving broader alternatives. For example:

- a fixed localized matched-subspace detector used as a default post-residual score head
- a guard-triggered adaptive detector that selectively enables whitening in clutter-heavy localized regions
- a fully whitened covariance-adaptive detector used in selected artifact-heavy or structurally bounded operating regimes
- a shrink-only penalty layer that cannot increase scores and that preserves a protected set
- exact-output-preserving batched execution using cached geometry and conditional branch execution

These implementations are illustrative best-known embodiments, not exclusive definitions of the invention.

## Suggested Claim Families For Later Nonprovisional Drafting

The current draft should support at least the following claim families in a later nonprovisional:

1. A computer-implemented method for generating an ultrasound score map from beamformed or residualized data using localized matched-subspace scoring.
2. A computer-implemented method in which a clutter-evidence metric selects between a non-whitened detector branch and a whitened detector branch.
3. A computer-implemented method in which a shrink-only penalty is applied to candidate regions while preserving a protected region.
4. A system claim covering processors, memory, and an ultrasound data interface configured to carry out the foregoing method steps.
5. A non-transitory computer-readable-medium claim covering instructions that cause a processor to carry out the foregoing method steps.

These claim families should be supported expressly in the filed provisional even if a formal claim set is not filed with the provisional application.

## Example Support For Later Claims

The provisional should support later claims along at least these families:

1. A method of localized matched-subspace scoring on post-residual ultrasound slow-time data.
2. A method of selective enabling of local whitening based on clutter evidence.
3. A method of generating a score map using fixed and whitened branches with adaptive switching.
4. A method of applying a shrink-only score penalty with a protected set and inert fallback mode.
5. A threshold-transfer method using a calibration bank and later held-out application.
6. A real-time execution method using batched tile-local processing and overlap-add aggregation.
7. A system and non-transitory computer-readable medium implementing the above.

## Filing and Scope Notes

This draft should not be treated as complete merely because related material exists in the paper or repository. The filed provisional should itself contain the operative disclosure and drawings needed to support later claims without relying on external links.

When converting this draft to a filed application, avoid narrowing statements such as “the invention is” followed by a single embodiment. Prefer “in some embodiments,” “in certain implementations,” or equivalent language, followed by meaningful alternatives.

Named benchmarks, datasets, frame counts, and hardware platforms from the manuscript should be disclosed only as examples unless a deliberate business decision is made to pursue narrower scope.

## Open Items For Counsel / Inventor Review

Before filing, confirm:

- exact inventorship
- ownership and assignment
- whether foreign filing is desired
- whether a government-interest statement is required
- whether the provisional should include one or more narrower fallback embodiments with more implementation detail
- whether to include explicit pseudo-code
- whether to include preliminary claim language despite claims not being required
- whether to include a separate drawing set prepared in patent style

## Repo Sources Supporting This Draft

Primary technical support:

- [stap_fus_methodology.tex](/home/arthu/stap-for-fus/stap_fus_methodology.tex)
- [provisional_patent_outline.md](/home/arthu/stap-for-fus/provisional_patent_outline.md)
- [pipeline/stap/temporal.py](/home/arthu/stap-for-fus/pipeline/stap/temporal.py)
- [pipeline/stap/temporal_shared.py](/home/arthu/stap-for-fus/pipeline/stap/temporal_shared.py)
- [sim/kwave/common.py](/home/arthu/stap-for-fus/sim/kwave/common.py)
- [sim/kwave/icube_bundle.py](/home/arthu/stap-for-fus/sim/kwave/icube_bundle.py)

Supporting figures:

- [figs/paper/stap_pipeline_bayes_block.pdf](/home/arthu/stap-for-fus/figs/paper/stap_pipeline_bayes_block.pdf)
- [figs/paper/hankelization_local_stap.pdf](/home/arthu/stap-for-fus/figs/paper/hankelization_local_stap.pdf)
- [figs/paper/doppler_band_geometry_psd.pdf](/home/arthu/stap-for-fus/figs/paper/doppler_band_geometry_psd.pdf)
- [figs/paper/shrink_only_tail_suppression.pdf](/home/arthu/stap-for-fus/figs/paper/shrink_only_tail_suppression.pdf)
