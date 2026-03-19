# Troubleshooting and Fit Guide

This page is for the first 10 minutes after a new lab clones the repo and asks:
"Will this actually fit our pipeline?"

## Can I Use This on My Data?

Yes, if you have:
- complex beamformed slow-time IQ, or
- a complex clutter-filtered residual cube from an existing SVD- or PD-style pipeline

Probably yes, with a quick adapter, if you have:
- beamformed IQ with axis order other than `(T, H, W)`
- an existing PD pipeline where the final readout can be swapped while leaving clutter filtering unchanged

Probably no, or not without additional research code, if you only have:
- rendered Doppler images
- magnitude-only frame stacks with no complex slow-time information
- sparse compounded frames intended for end-to-end PD reconstruction
- raw-channel data and you need this repo to beamform first

## The Public API Contract

The stable public API expects:
- `numpy` complex-valued input
- shape `(T, H, W)`
- the correct slow-time PRF in Hz
- a same-residual comparison mindset when evaluating variants

The detector is designed to answer:
- given this clutter-filtered residual, does changing only the final scoring rule reduce artifact leakage?

It is not designed to answer:
- what is the best clutter filter for this dataset?
- how do I reconstruct power Doppler from sparse compounded frames?
- how do I beamform from raw channel RF?

## Common Integration Mistakes

### 1. Wrong axis order

Symptom:
- output maps look wrong, scrambled, or unreasonably flat

Check:
- the public API expects `(T, H, W)`, not `(H, W, T)`

Fix:
- transpose before calling the detector

```python
residual_cube = np.transpose(residual_cube, (2, 0, 1))
```

### 2. Real-valued or magnitude-only input

Symptom:
- results are weak, unstable, or obviously inconsistent with the paper

Why:
- the detector relies on slow-time complex structure

Fix:
- use the complex IQ or complex clutter-filtered residual, not a rendered Doppler image

### 3. PRF is missing or wrong

Symptom:
- the detector underperforms or behaves inconsistently across acquisitions

Why:
- the Doppler bands are defined in Hz, so the wrong `prf_hz` shifts the meaning of the flow and guard bands

Fix:
- pass the actual slow-time PRF from acquisition metadata

### 4. Comparing different residuals

Symptom:
- it is unclear whether a gain comes from the detector or from upstream filtering changes

Why:
- the paper’s core discipline is same-residual comparison

Fix:
- hold the clutter-filtered residual fixed and change only the final readout when benchmarking variants

### 5. Ensemble too short for the chosen settings

Symptom:
- unstable scores, empty maps, or unintuitive behavior

Why:
- very short slow-time windows limit what any band-limited detector can resolve cleanly

Fix:
- start with the default `fixed` variant and representative windows similar to those already used in your pipeline

### 6. Expecting a universal improvement on real data

Symptom:
- the detector does not produce a dramatic visual improvement on the first real-IQ case

Why:
- the paper’s strongest evidence is the held-out SIMUS benchmark
- the real-IQ ULM audit is bounded and modest by design

Fix:
- calibrate expectations
- start with same-residual comparisons against your own PD baseline
- use the adaptive telemetry before escalating to full whitening

## Which Variant Should I Try First?

- `fixed`: default and recommended first integration target
- `adaptive`: use when you want guard telemetry and selective whitening without changing the default path everywhere
- `whitened`: test only when representative windows stay clutter-dominant and you can afford the extra cost
- `whitened_power`: diagnostic/ablation only

See also:
- [Integration Guide](integration.md)
- [Minimal API example](../examples/minimal_integration.py)
- [PD-to-detector swap example](../examples/svd_pipeline_readout_swap.py)

## If Nothing Looks Right

Start with this checklist:
1. Confirm the input is complex and shaped `(T, H, W)`.
2. Confirm `prf_hz` from metadata.
3. Run the minimal example to make sure the environment is healthy.
4. Run a same-residual comparison against your current PD readout.
5. Start with `fixed`, not `whitened`.

If you still think the detector is mismatched to your data, that is useful information rather than failure. The intended collaboration path is to test exactly that question on a fixed residual stream.

Contact: `arthur@skymesasystems.com`
