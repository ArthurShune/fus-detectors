# Provisional Patent Outline

This is a technical drafting outline for a U.S. provisional application based on the current invention as implemented and described in this repo. It is not legal advice. Before filing, have patent counsel review inventorship, assignment, domestic and foreign filing strategy, and claim scope.

## 1. Core Drafting Rule

Do not file the manuscript verbatim.

The provisional should be a patent-style specification organized around invention families, practical implementations, and alternative embodiments. The paper and benchmark suite are evidence that the detector works; they are not the invention itself.

The filing should read as a computer-implemented ultrasound imaging method and system that:

- receives beamformed ultrasound slow-time data or residualized slow-time data
- performs localized matched-subspace flow detection
- optionally enables local whitening when clutter evidence warrants it
- optionally applies a shrink-only penalty layer
- produces score maps, detection maps, vascular maps, or downstream decisions

That framing matters for both priority support under `35 U.S.C. 112(a)` and for presenting the invention as a practical application rather than only as a mathematical statistic.

## 2. Filing Facts To Respect Before The Preprint

Official USPTO points worth following:

- A provisional application can establish an early filing date, but it is not examined.
- A corresponding nonprovisional must be filed within 12 months to keep the benefit of the provisional filing date.
- A provisional should include a detailed written description satisfying `35 U.S.C. 112(a)` and any drawings needed to understand the invention.
- A provisional does not require formal claims, an oath, or an IDS, but it does require the correct filing paperwork and fee.
- The U.S. has grace-period rules for some inventor-originated disclosures, but many foreign jurisdictions do not. If foreign filing matters, file before public disclosure.

Official sources:

- USPTO provisional filing overview: <https://www.uspto.gov/patents/basics/apply/provisional-application>
- USPTO MPEP on provisional content: <https://www.uspto.gov/web/offices/pac/mpep/documents/0600_601_01_d.htm>
- USPTO subject-matter eligibility guidance entry point: <https://www.uspto.gov/web/offices/pac/mpep/s2106.html>
- USPTO Patent Center / online filing: <https://www.uspto.gov/patents/apply/patent-center>
- USPTO guidance on disclosure timing and foreign rights: <https://www.uspto.gov/patents/basics/international-protection/filing-patents-abroad>
- USPTO provisional drafting presentation: <https://www.uspto.gov/sites/default/files/documents/provisional-applications-6-2023.pdf>

## 3. What The Provisional Should Actually Cover

The patent should not be framed as “the paper” or “the benchmark suite.” The invention is the detector platform and deployment workflow, not the evaluation narrative.

The strongest package is:

1. A broad method, system, and non-transitory computer-readable-medium disclosure for a post-residual localized matched-subspace detector for beamformed ultrasound slow-time data.
2. A second layer covering selective local whitening, where whitening is activated only in tiles, pixels, or regions that satisfy a clutter-evidence rule.
3. A third layer covering the optional shrink-only penalty map with a protected set and fail-safe inert mode.
4. A fourth layer covering deployment-oriented thresholding and calibration transfer, if protection is wanted on the practical operating workflow as well as the score computation.
5. A fifth layer covering exact-output-preserving real-time GPU or accelerator execution, if commercial deployment speed matters.

## 4. Suggested Provisional Title

Primary title:

- `Systems And Methods For Localized Matched-Subspace Detection In Ultrasound Doppler Imaging`

Broader alternative titles:

- `Systems And Methods For Localized Post-Residual Flow Detection In Beamformed Ultrasound Data`
- `Systems And Methods For Adaptive Local Whitening And Matched-Subspace Scoring In Ultrafast Doppler Imaging`
- `Systems And Methods For Localized Flow-Consistent Signal Detection In Beamformed Ultrasound Data`

Use a broader ultrasound-facing title unless there is a deliberate reason to limit the filing to functional ultrasound only.

## 5. Core Invention Statement

Draft this early and clearly:

> The invention relates to systems and methods for detecting flow-consistent signal in beamformed ultrasound slow-time data after clutter suppression. A residual signal is analyzed locally using tile-based band-limited matched-subspace scoring rather than conventional power Doppler alone. In some embodiments, local whitening is selectively enabled when guard-band or related clutter-evidence features indicate structured nuisance. In some embodiments, an optional shrink-only penalty layer suppresses high-risk detections without increasing any score and while preserving protected flow-proxy regions. The disclosed methods improve nuisance discrimination, strict-tail behavior, and deployment practicality in functional ultrasound and ultrafast Doppler imaging.

## 6. Six Invention Families To Disclose

### A. Core localized matched-subspace detector

This is the anchor invention and should be described broadly.

Disclose:

- beamformed or otherwise formed complex slow-time ultrasound data
- optional baseline residualization or clutter suppression
- spatial tiling or regional partitioning
- tile-local slow-time vector extraction
- flow-band, guard-band, alias-band, and clutter-band operators or equivalent subspaces
- matched-subspace score using band-limited energy, optionally self-normalized
- overlap-add or other aggregation back to a pixel map

Key point to make explicit:

- the inventive change is in the downstream detector head after residualization, not necessarily in the upstream clutter filter

### B. Selective local whitening / adaptive branch

This is likely the strongest secondary claim family.

Disclose:

- a default non-whitened detector path
- a whitened detector path using local covariance adaptation
- a gating feature based on guard-band energy, alias-to-flow ratio, motion indicator, covariance conditioning, or other clutter evidence
- switching, promotion, interpolation, blending, or mixed execution between fixed and whitened branches
- switching at pixel level, tile level, block level, region level, or frame level

Make this broad:

- do not tie it only to one guard-band fraction formula
- cover features derived from energy, covariance, telemetry, motion, side-information, or nuisance-risk

### C. Fully whitened covariance-adaptive branch

Disclose:

- local covariance estimation from tile-local training support
- SCM, Tyler, Huber, trimmed covariance, shrinkage covariance, diagonal loading, eigenvalue flooring, or other conditioned covariance
- whitening followed by matched-subspace detection, power scoring, GLRT-style variants, or self-normalized score variants
- training-support strategies including Hankelized, pooled, subsampled, capped, or stride-selected support

### D. Shrink-only penalty layer

This is a distinct invention family and should be disclosed even if it is not the paper’s main story.

Disclose:

- side-information features computed from the same local data or auxiliary maps
- candidate set and protected set logic
- a strictly non-increasing mapping applied to scores
- inert-mode behavior when evidence is weak, missing, flat, or out of regime
- protected flow-proxy or vessel-proxy regions that remain unpenalized
- use of alias-band, guard-band, and flow-support features to drive shrinkage

This part is worth protecting because it is a clean safety mechanism:

- it reduces scores only
- it preserves a protected set
- it fails safely to “off”

### E. Threshold-transfer and deployment workflow family

This should be included if protection is wanted around practical deployment, not only around the score computation.

Disclose:

- calibration of thresholds from negative-only banks or separate calibration sets
- transfer of thresholds across windows, blocks, animals, sessions, devices, or scanners without per-window retuning
- right-tail score calibration using background or nuisance sets
- calibration for fixed FPR or matched operating points
- adaptation of thresholds across detector branches or variants
- output to display, storage, alerting, downstream analytics, or control logic

Do not oversell this family as a solved clinical workflow. Disclose it as a practical operating embodiment with multiple variants.

### F. System and exact-output-preserving real-time execution family

This family is worth including if the commercial story depends on producing the disclosed detector outputs within acquisition-time constraints.

Disclose:

- scanner-integrated, workstation, GPU, edge, software-only, and medium embodiments
- batched tile extraction and localized score computation
- conditional execution of whitening only for selected tiles, regions, pixels, or branch-promoted supports
- exact-output-preserving cached geometry terms, overlap counts, projector terms, or reusable localized operators
- fixed-batch or graph-replay execution strategies that stabilize repeated localized workloads
- overlap-add or equivalent stitching of localized outputs into a score map
- latency-sensitive modes that omit nonessential diagnostics or telemetry while preserving the same detector outputs
- exact-output-preserving fallback or bypass logic for localized support conditions, including clean-scene fast paths where appropriate

The point is not merely “use a GPU.” The point is a concrete execution architecture for the localized detector family that preserves the intended detector output while meeting acquisition-time budgets.

## 7. What Not To Lock To

Do not hard-code the filing to the paper’s current evaluation or implementation choices. These are embodiments only:

- MC-SVD or any other single residualizer
- Gammex, SIMUS/PyMUST, ULM, PALA, Shin, or any named dataset
- 32-frame, 64-frame, or 128-frame windows
- any one tile size, stride, or covariance recipe
- a single guard statistic or threshold formula
- RTX 4080, CUDA graphs, or any one hardware platform
- any one benchmark or figure from the manuscript

The specification should present these as examples in certain embodiments, followed immediately by alternatives.

## 8. What Not To Make Central

Do not make these the core invention:

- the benchmark suite
- the paper narrative
- specific datasets
- the exact numbers in the paper
- reviewer-facing terminology

These should be supporting examples only.

## 9. Suggested Section Outline For The Provisional

### Title

Use one of the titles above.

### Applicant details and filing cover information

- inventor names and residences
- applicant or assignee
- correspondence address
- any government-interest statement, if applicable

### Field

- ultrasound signal processing
- ultrafast Doppler
- functional ultrasound
- post-residual flow detection

### Background

Keep this measured and practical:

- conventional clutter suppression often leaves structured nuisance
- power-based downstream readouts can mis-rank nuisance as flow
- local residual heterogeneity creates a downstream detector problem
- low-false-positive operating regions make rare nuisance especially damaging

### Summary

Use separate short paragraphs for:

- fixed detector family
- adaptive detector family
- fully whitened detector family
- shrink-only penalty family
- threshold-transfer family
- exact-output-preserving real-time execution family
- system and hardware embodiments

### Brief Description Of Drawings

Include one paragraph per figure.

### Detailed Description

Recommended subsections:

1. Definitions and construction notes
2. Input data and baseline residualization
3. Tile extraction and temporal support formation
4. Doppler band geometry
5. Fixed matched-subspace detector
6. Local covariance estimation and whitening
7. Adaptive switching between fixed and whitened branches
8. Score aggregation
9. Optional shrink-only penalty layer
10. Threshold calibration and transfer
11. Exact-output-preserving real-time execution
12. System and medium embodiments
13. Example operating settings
14. Alternative embodiments

### Optional claim appendix

Claims are not required for a provisional, but including a draft claim appendix is useful because it forces the disclosure to support the scope you actually want later.

## 10. Drawings To Include

At minimum, include these figures or patent-clean redrawings of them:

1. End-to-end method overview
   - based on [figs/paper/stap_pipeline_bayes_block.pdf](/home/arthu/stap-for-fus/figs/paper/stap_pipeline_bayes_block.pdf)

2. Tile extraction / temporal embedding / covariance formation
   - based on [figs/paper/hankelization_local_stap.pdf](/home/arthu/stap-for-fus/figs/paper/hankelization_local_stap.pdf)

3. Doppler band geometry
   - based on [figs/paper/doppler_band_geometry_psd.pdf](/home/arthu/stap-for-fus/figs/paper/doppler_band_geometry_psd.pdf)

4. Adaptive switching logic
   - new drawing recommended: fixed path, clutter-evidence trigger, whitened path, merged output

5. Shrink-only penalty logic
   - new drawing recommended: candidate set, protected set, weight map, inert-mode sentinel

6. Threshold-transfer workflow
   - calibration bank, fixed threshold family, held-out deployment data

7. Real-time processing architecture
   - tile batch extraction, conditional localized execution, overlap-add, optional GPU or accelerator execution

8. System hardware architecture
   - scanner-integrated embodiment, workstation embodiment, edge embodiment

9. Example output maps
   - one or two clean examples are enough to support utility

The drawings do not need journal polish. They do need clarity and technical completeness.

## 11. Claim-Style Coverage To Seed In The Provisional

Even though claims are not required, write the disclosure so later claims can be supported across broad, medium, and narrow scope.

Primary independent claim directions:

1. `Method claim`
   - localized matched-subspace scoring on post-residual ultrasound slow-time data

2. `Adaptive method claim`
   - branch selection between fixed and whitened detection using clutter evidence

3. `Penalty-layer claim`
   - shrink-only post-score suppression with a protected set and inert fallback

4. `Real-time execution claim`
   - exact-output-preserving batched localized processing with conditional branch execution and reusable localized operators

5. `System claim`
   - processor, memory, and ultrasound-data interface configured to perform the method

6. `Non-transitory computer-readable-medium claim`
   - software instructions causing a processor to perform the method

Dependent support themes:

- beamformed IQ input vs residual-cube input
- tile geometry and overlap
- with and without temporal embedding or Hankelization
- different band constructions
- ratio scores vs energy scores vs GLRT-style scores
- covariance estimators and conditioners
- hard switch vs soft switch vs blended switch
- protected-set construction
- calibration-bank and threshold-transfer rules
- conditional execution and reusable localized operators
- hardware embodiment
- fallback and safe-disable logic

## 12. Definitions To Put In The Specification

The specification should explicitly define the special terms that later claim construction may depend on. At minimum:

- beamformed IQ
- residual data or residual cube
- tile or localized support region
- slow-time vector or slow-time snapshot
- flow band
- guard band
- alias band
- matched-subspace score
- fixed detector
- fully whitened detector
- adaptive detector
- candidate set
- protected set
- shrink-only penalty
- score map

Do not assume the paper’s notation is enough. Put the operative definitions into the provisional itself.

## 13. Specific Technical Alternatives To Enumerate

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
  - multiplicative shrinkage, capped suppression, monotone non-increasing transforms
- different deployment calibrations:
  - per-device, per-probe, per-session, separate-bank transfer, online recalibration, nuisance-class calibration
- different hardware:
  - CPU, GPU, FPGA, ASIC, edge device, scanner backend

## 14. Concrete Repo Material To Reuse In Drafting

Strongest source material:

- [stap_fus_methodology.tex](/home/arthu/stap-for-fus/stap_fus_methodology.tex)
- [provisional_patent_draft_specification.md](/home/arthu/stap-for-fus/provisional_patent_draft_specification.md)
- [pipeline/stap/temporal.py](/home/arthu/stap-for-fus/pipeline/stap/temporal.py)
- [pipeline/stap/temporal_shared.py](/home/arthu/stap-for-fus/pipeline/stap/temporal_shared.py)
- [sim/kwave/common.py](/home/arthu/stap-for-fus/sim/kwave/common.py)
- [sim/kwave/icube_bundle.py](/home/arthu/stap-for-fus/sim/kwave/icube_bundle.py)

Useful drawing sources:

- [figs/paper/stap_pipeline_bayes_block.pdf](/home/arthu/stap-for-fus/figs/paper/stap_pipeline_bayes_block.pdf)
- [figs/paper/hankelization_local_stap.pdf](/home/arthu/stap-for-fus/figs/paper/hankelization_local_stap.pdf)
- [figs/paper/doppler_band_geometry_psd.pdf](/home/arthu/stap-for-fus/figs/paper/doppler_band_geometry_psd.pdf)
- [figs/paper/shrink_only_tail_suppression.pdf](/home/arthu/stap-for-fus/figs/paper/shrink_only_tail_suppression.pdf)

Do not paste these verbatim. Convert them into patent-style prose with alternatives, ranges, and multiple embodiments.

## 15. What To Add That Is Not Strong Enough In The Paper

The manuscript is good source material, but the provisional should be broader than the paper in these ways:

- define special terms explicitly
- describe the invention as a practical process and system, not just a score
- enumerate alternatives for every branch and score form
- describe broader calibration and transfer embodiments
- describe broader hardware embodiments
- include safe-failure and fallback behavior
- include medium and system embodiments

## 16. Do Not Rely On The Repo Or Preprint To Supply Missing Disclosure

Do not assume a repository link, a future supplement, or the manuscript itself will cure missing disclosure. The filed specification and drawings should contain the operative technical description needed to support later claims. If a later claim depends on details that were not in the filed provisional, those details may be treated as new matter and lose the earlier filing date.

## 17. Inventorship And Ownership Checklist

Before filing, confirm:

- who conceived the core fixed detector family
- who conceived the adaptive switching family
- who conceived the shrink-only penalty family
- who conceived the threshold-transfer or deployment family
- whether assignment to SkyMesa Systems Inc. is required before filing
- whether any government-interest statement is required

Inventorship should be mapped to the claim families expected later, not just to paper authorship.

## 18. Practical Filing Mechanics

- File before the preprint if foreign filing optionality matters.
- Use Patent Center or the current USPTO provisional filing workflow.
- Include the required cover sheet or ADS information.
- Include drawings up front if they are needed to understand the invention.
- Watch sheet count and size fees if tempted to dump the manuscript and appendix into the filing.
- Keep a dated internal record of the exact provisional package that was filed.

## 19. Recommended Filing Package

Best near-term package:

1. `Patent-style provisional specification`
   - broad method/system description with alternatives

2. `Patent-clean figure set`
   - 8 to 12 black-and-white line drawings are enough

3. `Inventorship and assignment checklist`
   - to hand to counsel before filing

4. `Optional claim appendix`
   - even though not required, it forces scope discipline

## 20. Practical Priority Order

If time is tight before the preprint:

1. File the core matched-subspace detector family.
2. Make sure selective whitening is fully disclosed.
3. Make sure the shrink-only penalty layer is fully disclosed.
4. Add the threshold-transfer embodiments.
5. Add the system and real-time execution embodiments.

If something must be cut for time, cut benchmark detail before cutting algorithm variants.

## 21. Recommended Attorney Hand-Off Note

When sending this to counsel, the short note should be:

> We want a provisional centered on a post-residual localized matched-subspace detector family for beamformed ultrasound slow-time data, with broad coverage on tile-local matched-subspace scoring, selective local whitening driven by clutter evidence, an optional shrink-only penalty layer with protected sets, threshold-transfer embodiments, and exact-output-preserving real-time batched implementations. The manuscript and code are supporting disclosure materials, but the filing should be drafted as a detector and deployment platform rather than as a benchmark paper.
