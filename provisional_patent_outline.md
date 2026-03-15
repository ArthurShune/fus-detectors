# Provisional Patent Outline

This is a technical drafting outline for a U.S. provisional application based on the current invention as implemented and described in this repo. It is not legal advice. Before filing, have a patent attorney review the scope, inventorship, assignments, and foreign-filing strategy.

## 1. Filing Facts To Respect Before The Preprint

Official USPTO points worth following:

- A provisional application is a placeholder that can establish an early filing date, but it is not examined. A corresponding nonprovisional must be filed within 12 months to keep the benefit of the provisional filing date.
- A provisional should include a detailed written description that satisfies `35 U.S.C. 112(a)` and any drawings needed to understand the invention.
- A provisional does not require claims, and it does not require an oath or declaration, but it does require a provisional cover sheet or ADS and the filing fee.
- The USPTO allows a one-year grace period after an inventor-originated public disclosure, but many other countries do not. If foreign filing matters, file before public disclosure.

Official sources:

- USPTO provisional filing overview: <https://www.uspto.gov/patents/basics/apply/provisional-application>
- USPTO MPEP on provisional content: <https://www.uspto.gov/web/offices/pac/mpep/documents/0600_601_01_d.htm>
- USPTO Patent Center / filing online: <https://www.uspto.gov/patents/apply/filing-online/submitting-patent-applications-or-patent-prosecution>
- USPTO on disclosure and foreign rights: <https://www.uspto.gov/patents/basics/international-protection/filing-patents-abroad>

## 2. What The Provisional Should Actually Cover

The patent should not be framed as "the paper" or "the benchmark suite." The invention is the detector system and deployment workflow, not the evaluation narrative.

The strongest filing package is:

1. A broad method/system/computer-readable-medium disclosure for a post-residual localized matched-subspace detector for beamformed ultrasound slow-time data.
2. A second layer covering selective local whitening, where whitening is activated only in tiles or regions that satisfy a clutter-evidence rule.
3. A third layer covering the optional shrink-only penalty map with a protected set and fail-safe inert mode.
4. A fourth layer covering deployment-oriented thresholding and calibration transfer, if you want protection on the practical operating workflow and not only on the score computation.
5. A fifth layer covering exact-output-preserving real-time GPU execution, if commercial deployment speed matters.

## 3. Suggested Provisional Title

Primary title:

- `Localized Matched-Subspace Detection For Functional Ultrasound And Ultrafast Doppler Imaging`

Broader alternative titles:

- `Systems And Methods For Localized Post-Residual Flow Detection In Beamformed Ultrasound Data`
- `Systems And Methods For Adaptive Local Whitening And Matched-Subspace Scoring In Ultrafast Doppler Imaging`

## 4. Core Invention Statement

Draft this early and clearly:

> The invention relates to systems and methods for detecting flow-consistent signal in beamformed ultrasound slow-time data after clutter suppression. A residual signal is analyzed locally, using tile-based band-limited matched-subspace scoring rather than conventional power Doppler alone. In some embodiments, local whitening is selectively enabled when guard-band or related clutter-evidence features indicate structured nuisance. In some embodiments, an optional shrink-only penalty layer suppresses high-risk detections without increasing any score and while preserving protected flow-proxy regions. The disclosed methods improve nuisance discrimination, strict-tail behavior, and deployment practicality in functional ultrasound and ultrafast Doppler imaging.

## 5. Invention Families To Disclose

### A. Core localized matched-subspace detector

This is the anchor invention and should be described broadly.

Disclose:

- beamformed or otherwise formed complex slow-time ultrasound data
- optional baseline residualization or clutter suppression
- spatial tiling or regional partitioning
- tile-local slow-time vector extraction
- flow-band, guard-band, alias-band, and clutter-band operators
- matched-subspace score using band-limited energy, optionally self-normalized
- overlap-add or other aggregation back to a pixel map

Key point to make explicit:

- the inventive change is in the downstream detector head after residualization, not necessarily in the upstream clutter filter

### B. Selective local whitening / adaptive branch

This is likely the most commercially valuable extension.

Disclose:

- a default non-whitened detector path
- a whitened detector path using local covariance adaptation
- a gating feature based on guard-band energy, alias-to-flow ratio, motion indicator, or other clutter evidence
- switching, promotion, interpolation, blending, or mixed execution between fixed and whitened branches
- switching at pixel level, tile level, block level, or region level

Make this broad:

- do not tie it only to one guard-band fraction formula
- cover any feature derived from energy, covariance, telemetry, motion, side-information, or nuisance-risk

### C. Fully whitened specialist branch

Disclose:

- local covariance estimation from tile-local training support
- SCM, Tyler, Huber, trimmed covariance, shrinkage covariance, diagonal loading, or other conditioned covariance
- whitening followed by matched-subspace detection, power scoring, GLRT-style variants, or self-normalized score variants
- different training-support strategies, including Hankelized, pooled, subsampled, capped, or stride-selected support

### D. Shrink-only penalty layer

This is a distinct invention family and should be disclosed even if not made central in the paper narrative.

Disclose:

- side-information features computed from the same local data or from auxiliary maps
- candidate set and protected set logic
- a strictly non-increasing mapping applied to scores
- inert-mode behavior when evidence is weak or out of regime
- protected flow-proxy or vessel-proxy regions that remain unpenalized
- use of alias-band, guard-band, and flow-support features to drive shrinkage

This part is worth protecting because it is a clean safety-oriented mechanism:

- it reduces scores only
- it preserves a protected set
- it fails safely to "off"

### E. Empirical tail-calibration / deployment thresholding workflow

This should be included if you want protection around practical deployment, not just score computation.

Disclose:

- calibration of thresholds from negative-only banks or separate calibration sets
- transfer of thresholds across windows, blocks, animals, or sessions without per-window retuning
- right-tail score calibration using background or nuisance sets
- calibration for fixed FPR or matched operating points
- adaptation of thresholds across detector branches or variants

Do not oversell this as solved. Just disclose the workflow and embodiments.

### F. Real-time batched execution architecture

This is optional but worth including if you may commercialize.

Disclose:

- GPU batching of tile extraction, temporal embedding, covariance, whitening, and scoring
- exact-output-preserving cached geometry operations
- conditional execution of whitening only in selected tiles
- graph-captured or replayed kernels
- overlap-add stitching
- latency-aware variants that omit nonessential diagnostics

This may support separate system claims later, even if the first nonprovisional focuses mainly on the detector itself.

## 6. What Not To Make Central

Do not make these the core invention:

- the benchmark suite
- the paper narrative
- specific datasets
- the exact numbers in the paper
- specific reviewer-facing terminology

These should be supporting examples only.

## 7. Suggested Section Outline For The Provisional

### Title

Use one of the titles above.

### Field

- ultrasound signal processing
- functional ultrasound
- ultrafast Doppler
- post-residual flow detection

### Background

Keep this short and practical:

- conventional clutter suppression often leaves structured nuisance
- PD/Kasai-style scoring can mis-rank nuisance as flow
- global methods do not address local residual heterogeneity well
- low-FPR use cases make rare nuisance especially damaging

### Summary

One to three pages that explain:

- fixed detector
- adaptive detector
- fully whitened detector
- optional shrink-only penalty
- deployment-oriented threshold calibration
- real-time implementation

### Brief Description Of Drawings

Include a one-paragraph description of each figure.

### Detailed Description

Recommended subsections:

1. Input data and baseline residualization
2. Tile extraction and temporal support formation
3. Doppler band geometry
4. Fixed matched-subspace detector
5. Local covariance estimation and whitening
6. Adaptive switching between fixed and whitened branches
7. Score aggregation
8. Optional shrink-only penalty layer
9. Threshold calibration and transfer
10. Real-time batched implementation
11. Example operating settings
12. Alternative embodiments

## 8. Drawings To Include

At minimum, include these figures or patent-clean redrawings of them:

1. End-to-end method overview
   - based on [figs/paper/stap_pipeline_bayes_block.pdf](/home/arthu/stap-for-fus/figs/paper/stap_pipeline_bayes_block.pdf)

2. Tile extraction / temporal embedding / covariance formation
   - based on [figs/paper/hankelization_local_stap.pdf](/home/arthu/stap-for-fus/figs/paper/hankelization_local_stap.pdf)

3. Doppler band geometry
   - based on [figs/paper/doppler_band_geometry_psd.pdf](/home/arthu/stap-for-fus/figs/paper/doppler_band_geometry_psd.pdf)

4. Adaptive switching logic
   - new drawing recommended: fixed path, guard-band trigger, whitened path, merged output

5. Shrink-only penalty logic
   - new drawing recommended: candidate set, protected set, weight map, inert-mode sentinel

6. Real-time processing architecture
   - new drawing recommended: tile batch extraction, local processing, overlap-add, optional GPU execution

7. Example output maps
   - one structural example is enough to support utility; do not overload with paper figures

The drawings do not need journal polish. They do need clarity and technical completeness.

## 9. Claim-Style Coverage To Seed In The Provisional

Even though claims are not required, write the disclosure so later claims can be supported across broad, medium, and narrow scope.

### Broad claim themes

- A method for processing beamformed ultrasound slow-time data by locally scoring residual signal with a matched-subspace detector.
- A method for detecting flow-consistent signal in ultrasound using regional band-limited matched-subspace scoring after clutter suppression.
- A system comprising a processor configured to compute tile-local matched-subspace scores and generate a detection map.
- A non-transitory computer-readable medium storing instructions for same.

### Medium-scope themes

- The method wherein the detector selectively enables local whitening based on a clutter-evidence feature.
- The method wherein the clutter-evidence feature is derived from guard-band energy, alias-band energy, motion metrics, covariance condition metrics, or combinations thereof.
- The method wherein the score is self-normalized by out-of-band or orthogonal-complement energy.
- The method wherein overlapping tiles are aggregated by overlap-add or weighted fusion.

### Narrower fallback themes

- A guard-band-triggered switch between fixed and whitened detector branches.
- A shrink-only post-score penalty map with a protected set and fail-safe inert mode.
- A covariance-estimation workflow using robust estimation and conditioning for local whitening.
- A fixed-threshold transfer workflow using a calibration bank learned on one set of windows and applied to held-out windows.
- A batched GPU pipeline for exact-output-preserving implementation of the detector.

## 10. Specific Technical Alternatives To Enumerate

The biggest provisional mistake is under-disclosing variants and then being unable to add them later without new-matter problems.

Explicitly enumerate:

- different residualizers:
  - SVD, adaptive-global SVD, local SVD, RPCA, HOSVD, no residualizer, learned residualizer
- different local supports:
  - tiles, superpixels, patches, regions of interest, multiscale tiles
- different time-support construction:
  - direct slow-time vectors, Hankel matrices, sliding windows, pooled snapshots
- different score forms:
  - power score, ratio score, GLRT-style score, matched-subspace ratio, whitened power
- different covariance estimators:
  - SCM, shrinkage, Tyler, Huber, trimmed covariance, Bayesian covariance, factorized covariance
- different switch rules:
  - thresholded, probabilistic, blended, hard-switched, learned but frozen
- different penalty maps:
  - multiplicative shrinkage, additive suppression, capped suppression, monotone non-increasing transforms
- different deployment calibrations:
  - per-device, per-probe, per-session, separate-bank transfer, online recalibration, nuisance-class calibration
- different hardware:
  - CPU, GPU, FPGA, ASIC, edge device, ultrasound scanner backend

## 11. Concrete Repo Material To Reuse In Drafting

Good starting sources:

- [stap_fus_methodology.tex](/home/arthu/stap-for-fus/stap_fus_methodology.tex)
  - introduction / method / score definitions / deployment-transfer language
- [figs/paper/stap_pipeline_bayes_block.pdf](/home/arthu/stap-for-fus/figs/paper/stap_pipeline_bayes_block.pdf)
- [figs/paper/hankelization_local_stap.pdf](/home/arthu/stap-for-fus/figs/paper/hankelization_local_stap.pdf)
- [figs/paper/doppler_band_geometry_psd.pdf](/home/arthu/stap-for-fus/figs/paper/doppler_band_geometry_psd.pdf)
- [sim/kwave/common.py](/home/arthu/stap-for-fus/sim/kwave/common.py)
- [pipeline/stap/temporal.py](/home/arthu/stap-for-fus/pipeline/stap/temporal.py)
- [pipeline/stap/temporal_shared.py](/home/arthu/stap-for-fus/pipeline/stap/temporal_shared.py)

These are especially important to support enablement:

- actual score formulas
- actual adaptive switching rule
- actual shrink-only logic
- actual real-time execution path

## 12. What I Would Put In The First Draft Now

If I were drafting the provisional this week, I would include:

- 1 to 2 pages of background
- 2 to 3 pages of summary
- 8 to 15 pages of detailed description
- 6 to 10 figures
- at least 2 pages enumerating variants and alternatives

That is enough for a serious provisional. It does not need to look like a journal paper. It does need to disclose enough technical detail that later claims are supported.

## 13. Practical Priority Order

If time is tight before the preprint:

1. File the core matched-subspace detector family.
2. Make sure selective whitening is fully disclosed.
3. Make sure the shrink-only penalty layer is fully disclosed.
4. Add the threshold-transfer embodiments.
5. Add the real-time GPU embodiments.

If something must be cut for time, cut benchmark detail before cutting algorithm variants.

## 14. Recommended Attorney Hand-Off Note

When sending this to counsel, the short note should be:

> We want a provisional centered on a post-residual localized matched-subspace detector family for beamformed fUS/ultrafast Doppler slow-time data, with broad coverage on tile-local matched-subspace scoring, selective local whitening driven by clutter evidence, an optional shrink-only penalty layer with protected sets, fixed-threshold transfer embodiments, and exact-output-preserving real-time batched implementations. The paper and code are supporting disclosure materials, but the patent should be drafted as a detector and deployment platform, not as a benchmark paper.
