# Draft Provisional Specification

## Title

Systems and Methods for Localized Subspace-Based Detection in Ultrasound Doppler Imaging

## Field

The present disclosure relates to ultrasound signal processing and, more particularly, to localized post-residual detection in ultrasound Doppler imaging, including functional ultrasound imaging, ultrafast Doppler imaging, power Doppler imaging, color flow imaging, microvascular flow detection, and related processing of beamformed complex slow-time data.

## Background

Ultrasound Doppler imaging systems commonly perform a clutter-suppression or residualization stage followed by a downstream flow readout, such as power Doppler, log-power Doppler, autocorrelation-based magnitude estimation, or related energy statistics. In many operating regimes, the clutter-suppression stage reduces dominant tissue or stationary clutter but does not fully eliminate structured nuisance in the residual signal.

Residual nuisance may include, for example, phase instability, aliased flow, specular leakage, vibration, probe motion, local nonstationary interference, or other structured artifacts. Conventional downstream power-based readouts can assign large responses to such nuisance, particularly in low-false-positive operating regimes or in scenes with substantial residual artifact burden.

Accordingly, there remains a need for a downstream detector framework that operates on beamformed or residualized slow-time ultrasound data, locally distinguishes target flow-consistent signal from non-target or nuisance signal, and supports deployment-oriented operating modes ranging from simple fixed scoring to covariance-adaptive localized scoring, optional bounded suppression behavior, and calibration workflows suitable for transfer across held-out data.

## Summary

The disclosure provides systems, methods, and non-transitory computer-readable media for localized detector processing in ultrasound Doppler imaging.

In some embodiments, beamformed ultrasound data or residualized slow-time ultrasound data are received and processed in localized spatial supports rather than only globally. The localized supports may comprise overlapping tiles, regions, patches, adaptive windows, superpixels, multiscale neighborhoods, or other localized support regions. For each localized support, the system forms one or more slow-time vectors or embedded slow-time snapshots and computes a detector score based on a target-aligned component relative to one or more non-target components. In some embodiments, the detector score is a matched-subspace score, subspace-energy score, projected-energy score, normalized ratio score, whitened-power score, generalized likelihood ratio style statistic, or other covariance-adaptive energy score.

In some embodiments, the detector employs a plurality of detector branches. A fixed branch computes a non-whitened or identity-whitened localized score. A covariance-adaptive branch estimates and conditions a localized covariance matrix, whitens localized data, and computes a covariance-adaptive score. An adaptive policy branch selects, blends, or otherwise combines two or more detector branches based on one or more clutter-evidence features, nuisance-risk features, motion indicators, covariance-condition indicators, support-quality indicators, or combinations thereof.

In some embodiments, the system further applies a shrink-only post-score transform. The shrink-only transform may define a candidate set eligible for score reduction, a protected set excluded from reduction, and an inert mode used when side-information is missing, flat, inconsistent, or otherwise non-actionable. In some embodiments, the shrink-only transform is constrained not to increase any detector score.

In some embodiments, detector thresholds are learned from calibration data, such as negative-only calibration banks, nuisance banks, separate sessions, separate subjects, separate devices, or other designated calibration corpora, and transferred to held-out or deployment data without per-window recalibration.

In some embodiments, the disclosure further provides exact-output-preserving execution architectures that reduce computation while preserving the detector outputs associated with the disclosed detector definition or branch-selection policy. Such embodiments may include batched tile extraction, temporal embedding, localized covariance estimation, conditional execution of covariance-adaptive processing, cached localized operators, overlap-add aggregation, graph replay, or hardware acceleration.

## Brief Description of the Drawings

1. **Figure 1: End-to-end processing overview.**
   A block diagram showing beamformed IQ input, residualization, localized detector branches, an optional shrink-only penalty, and output map generation.

2. **Figure 2: Localized support extraction and aggregation.**
   A schematic showing extraction of localized spatial supports from a residual slow-time cube, formation of slow-time vectors or embedded snapshots, per-support detector scoring, and aggregation of local results.

3. **Figure 3: Adaptive branch policy.**
   A diagram showing clutter-evidence features controlling selection or blending between a fixed detector branch and a covariance-adaptive detector branch.

4. **Figure 4: Shrink-only suppression layer.**
   A diagram showing candidate-set logic, protected-set logic, side-information features, and shrink-only score suppression.

5. **Figure 5: Threshold calibration and transfer.**
   A diagram showing threshold learning from calibration data and application of learned thresholds to held-out deployment data without per-window recalibration.

6. **Figure 6: Exact-output-preserving execution architecture.**
   A system diagram showing reusable localized operators, conditional branch execution, batched score generation, and overlap-add output formation.

## Detailed Description

### High-Level Embodiment Overview

In one embodiment, an upstream beamforming stage and an optional residualization stage provide a residual slow-time data structure to a localized detector head. The localized detector head forms one or more localized supports, generates one or more slow-time vectors or embedded snapshots for each support, applies one or more target and non-target operators or subspaces, computes one or more localized scores, and aggregates the localized scores into an output map. The output map may be thresholded, displayed, further analyzed, or combined with an optional shrink-only suppression layer. The resulting output may comprise a score map, detection map, vascular map, heat map, risk map, monitoring map, or other downstream representation.

### Ordered End-to-End Method Embodiment

In one embodiment, a processor performs the following ordered operations:

1. receives beamformed ultrasound data or residualized slow-time ultrasound data;
2. optionally applies a clutter-suppression or residualization stage;
3. defines a plurality of localized support regions;
4. forms one or more slow-time vectors or embedded snapshots for each localized support region;
5. defines one or more target and non-target operators or subspaces for the localized support;
6. computes, for at least some localized supports, a fixed localized detector score;
7. computes, for at least some localized supports, a covariance-adaptive localized detector score based on a localized covariance estimate and whitening transform;
8. selects, blends, or otherwise combines localized detector branches according to one or more clutter-evidence or nuisance-risk features;
9. aggregates localized detector scores to form an output map;
10. optionally applies a shrink-only suppression transform to at least a subset of the detector scores or aggregated outputs;
11. optionally applies one or more thresholds derived from calibration data; and
12. generates a score map, detection map, vascular map, heat map, risk map, monitoring map, or other output.

### Input and Residualization

In some embodiments, the input comprises beamformed in-phase and quadrature ultrasound data arranged over time and space. In some embodiments, the input comprises residualized slow-time data after a clutter-suppression stage. The residualization stage may comprise singular-value decomposition, adaptive singular-value decomposition, blockwise local singular-value decomposition, robust principal-component decomposition, higher-order singular-value decomposition, learned clutter suppression, temporal filtering, low-rank filtering, motion-compensated filtering, or combinations thereof.

In some embodiments, the residualization stage is implemented independently of the disclosed detector head such that multiple downstream detector heads may operate on the same residual input.

### Localized Support Formation

In some embodiments, the detector processes data in overlapping tiles. In other embodiments, the detector processes regions, patches, adaptive windows, superpixels, multiscale neighborhoods, or other localized supports. For each support, one or more slow-time vectors may be formed directly from temporal samples, pooled across a spatial support, or derived from sliding-window temporal embedding. In some embodiments, a Hankel or similar embedding generates multiple snapshots from one or more localized supports.

### Target and Non-Target Operators

In some embodiments, the detector uses one or more target and non-target subspace definitions or equivalent operators. Example operators include a target flow band, a clutter or direct-current band, a guard band, an alias or high-velocity nuisance band, and one or more out-of-band complements. The operators may be implemented as exact projectors, approximate projectors, filters, analysis operators, Fourier-domain selection operators, wavelet-domain operators, or other subspace representations. In some embodiments, operator definitions are fixed for an operating regime. In other embodiments, they are adapted according to acquisition settings or metadata.

### Fixed Detector Embodiment

In some embodiments, a fixed detector branch computes a localized non-whitened or identity-whitened score. For a localized vector `x`, an example score family is:

`S_fixed = flow_band_energy / (complement_energy + rho * total_energy + epsilon)`

where `rho` and `epsilon` are stabilization terms. Other fixed score formulations may be used, provided that the score compares target-aligned signal evidence to one or more non-target components. In some embodiments, the fixed detector is used as a default branch in lower-risk or lower-compute regimes.

### Covariance-Adaptive Detector Embodiment

In some embodiments, a covariance-adaptive branch estimates a localized covariance matrix from one or more training snapshots associated with a localized support. The covariance estimate may comprise a sample covariance matrix, shrinkage covariance, robust M-estimator, Tyler estimator, Huber estimator, trimmed covariance, diagonally loaded covariance, or combinations thereof. In some embodiments, the covariance estimate is conditioned using shrinkage, diagonal loading, trace normalization, eigenvalue flooring, condition-number targeting, or combinations thereof.

A whitening transform is derived from the conditioned covariance estimate and applied before detector scoring. The resulting score may comprise a matched-subspace ratio, whitened-power score, generalized likelihood ratio style statistic, or other covariance-adaptive energy score.

In one example, a covariance matrix is estimated from tile-local embedded snapshots, diagonally loaded to avoid ill-conditioning, normalized to a target trace level, and whitened before computing a target-versus-complement energy statistic.

### Adaptive Detector Policy

In some embodiments, the system computes one or more clutter-evidence features for a localized support and uses those features to control detector-branch execution. Example features include guard-band energy, guard-band fraction, alias-band energy, alias-to-flow ratio, motion magnitude, registration residual, covariance-condition metrics, support-quality metrics, nuisance-risk indicators, or combinations thereof.

In some embodiments, the detector policy is a hard switch between a fixed branch and a covariance-adaptive branch. In some embodiments, the policy is a soft weighting, confidence-weighted combination, or multilevel selection among more than two branches. In some embodiments, the policy is applied per support, per pixel, per region, per frame, or per operating regime.

In one concrete example, a guard-band fraction is computed for each localized support. When the guard-band fraction exceeds a threshold, the covariance-adaptive branch is enabled for that support; otherwise the fixed branch is used. In another example, the guard-band fraction and alias-to-flow ratio are combined into a branch-confidence score that blends the fixed and covariance-adaptive outputs.

### Aggregation and Output Maps

Localized scores may be aggregated using overlap-add, weighted averaging, confidence weighting, interpolation, voting, or related aggregation procedures. In some embodiments, branch-specific outputs are aggregated separately and then combined. In some embodiments, all supports overlapping a given spatial location contribute to that location's final output.

The output may comprise a score map, detection map, vascular map, heat map, risk map, monitoring map, thresholded display, or downstream analysis input.

### Shrink-Only Suppression

In some embodiments, the system applies a shrink-only transform to detector scores. Side-information features used by the transform may include flow-band energy, guard-band energy, alias-band energy, alias-to-flow ratios, vessel-proxy support, flow-proxy coverage, local background indicators, spatial support consistency, motion indicators, clutter-risk indicators, or combinations thereof.

In some embodiments, the shrink-only logic defines a candidate set eligible for reduction and a protected set excluded from reduction. In some embodiments, the transform is multiplicative with a weight satisfying `0 < w <= 1`. In other embodiments, the transform is any monotone non-increasing transform. In some embodiments, if side-information is non-actionable, missing, flat, or inconsistent, the transform defaults to an inert mode.

In one concrete example, a candidate set comprises supports having elevated alias-to-flow ratio or elevated guard-band fraction, while a protected set comprises supports having high flow-band coverage and spatial consistency with neighboring supports. If both the candidate-set evidence and protected-set evidence are weak or contradictory, the shrink-only layer remains inert.

### Integrity Controls and Safe Failure Modes

In some embodiments, the system detects invalid band discretization, missing or inconsistent metadata, insufficient support quality, proxy-mask circularity, or execution conditions requiring fallback behavior. In some embodiments, the system disables adaptive processing or shrink-only suppression when its prerequisites are not satisfied. In some embodiments, fallback to the fixed detector branch is treated as a valid operational mode.

### Threshold Calibration and Transfer

In some embodiments, one or more thresholds are learned from calibration data, such as negative-only calibration banks, nuisance banks, separate sessions, separate subjects, separate devices, separate blocks, or held-out windows. Thresholds may be branch-specific, regime-specific, or shared across multiple branches.

In one concrete example, a threshold is learned from a calibration bank containing only negative or nuisance-rich windows and then applied without per-window recalibration to held-out deployment windows from a different session. In another example, separate thresholds are learned for a fixed branch and a covariance-adaptive branch and then transferred unchanged to held-out blocks acquired with the same pulse-repetition-frequency regime.

### Exact-Output-Preserving Real-Time Execution

In some embodiments, runtime performance is improved using batched tile extraction, tensor-unfold procedures, batched temporal embedding, batched covariance estimation, cached overlap counts, cached geometry terms, cached projector terms, conditional branch execution, graph replay, overlap-add reconstruction, or hardware acceleration, while preserving the detector outputs associated with the disclosed detector definition or policy.

In one concrete example, localized projector terms and overlap counts are cached for repeated windows, temporal embedding is executed in fixed-size batches, and covariance-adaptive scoring is executed only for supports selected by the branch policy. Supports remaining on the fixed branch bypass covariance estimation and whitening while preserving the same detector outputs that would result from the disclosed adaptive policy.

### System and Medium Embodiments

In some embodiments, a system comprises an ultrasound data interface configured to receive beamformed IQ data or residualized slow-time data, one or more processors, memory storing instructions, and one or more displays or output interfaces. The system may be embodied in an ultrasound scanner, ultrasound research platform, workstation, portable device, edge device, remote server, or cloud-connected processing system.

In some embodiments, a non-transitory computer-readable medium stores instructions that, when executed by one or more processors, cause the one or more processors to perform any combination of the operations disclosed herein, including localized support formation, slow-time vector formation, target and non-target operator application, fixed detector scoring, covariance-adaptive detector scoring, adaptive policy execution, aggregation, shrink-only suppression, threshold calibration or application, integrity control, fallback behavior, logging, and output generation.

### Alternative Embodiments

The disclosed techniques are not limited to any particular residualizer, dataset, benchmark, frame count, support size, pulse-repetition frequency, covariance recipe, hardware platform, or evaluation protocol. In some embodiments, the detector is applied to beamformed or residualized slow-time data. In some embodiments, additional or alternative subspace constructions are used. In some embodiments, more than two detector branches are available and their outputs are selected or combined under a policy responsive to nuisance or clutter evidence.

In some embodiments, the shrink-only transform is applied before or after aggregation. In some embodiments, one or more reference maps, anatomical priors, or spatial context features provide side information. In some embodiments, learned nuisance features or learned branch policies are used as optional policy inputs.

### Example Use Cases

The disclosed techniques may be used in functional ultrasound imaging, ultrafast Doppler imaging, microvascular imaging, bedside imaging, mobile imaging, intra-operative imaging, structural or diagnostic imaging workflows, monitoring applications, interventional workflows, or other applications in which residual nuisance after clutter suppression can distort downstream interpretation or operating-point stability.

### Concluding Breadth-Preserving Paragraph

The foregoing embodiments are illustrative and not limiting. Features described in connection with one embodiment may be combined with features of another embodiment unless the context clearly indicates otherwise. The disclosed subject matter is not limited to any particular residualization technique, detector formula, covariance estimator, support geometry, operator basis, branch count, threshold source, hardware platform, or output representation. Equivalent localized support definitions, target and non-target operator constructions, branch-control policies, monotone non-increasing suppression transforms, aggregation procedures, and execution architectures may be used while remaining within the scope of the disclosed systems and methods.

## Suggested Claim Families For Later Nonprovisional Drafting

1. A computer-implemented method for generating an ultrasound score map from beamformed or residualized data using localized subspace-based scoring.
2. A computer-implemented method in which a clutter-evidence metric selects between a fixed detector branch and a covariance-adaptive detector branch.
3. A computer-implemented method in which a shrink-only penalty is applied to candidate regions while preserving a protected region.
4. A computer-implemented real-time execution method using conditional localized execution, reusable localized operators, and batched aggregation while preserving detector outputs.
5. A system claim covering processors, memory, and an ultrasound data interface configured to carry out the foregoing method steps.
6. A non-transitory computer-readable-medium claim covering instructions that cause a processor to carry out the foregoing method steps.

## Example Support For Later Claims

1. A method of localized subspace-based scoring on post-residual ultrasound slow-time data.
2. A method of selective enabling of local whitening based on clutter evidence.
3. A method of generating a score map using fixed and covariance-adaptive branches with adaptive selection.
4. A method of applying a shrink-only score penalty with a protected set and inert fallback mode.
5. A threshold-transfer method using a calibration bank and later held-out application.
6. A real-time execution method using batched tile-local processing, conditional branch execution, reusable localized operators, and overlap-add aggregation.
7. A system and non-transitory computer-readable medium implementing the above.
