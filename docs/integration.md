# Integration Guide

`fus-detectors` is easiest to integrate when your existing pipeline already
produces a complex clutter-filtered slow-time cube with shape `(T, H, W)`.
The public API keeps that contract narrow on purpose: you hand it the residual
cube and the slow-time PRF, and it returns a PD-style detector readout plus the
primary right-tail score map.

## Stable Import Surface

```python
from fus_detectors import DetectorConfig, score_residual_cube
```

Use the stable `fus_detectors` package for external integration. Internal
modules under `pipeline/` remain available for research work, but they are not
the compatibility layer intended for third-party pipelines.

For a runnable command-line example, see
[`examples/minimal_integration.py`](../examples/minimal_integration.py).
For a more realistic “swap the final readout in an existing SVD pipeline”
pattern, see [`examples/svd_pipeline_readout_swap.py`](../examples/svd_pipeline_readout_swap.py).
If you are not sure whether your data fit this contract, start with
[`docs/troubleshooting.md`](troubleshooting.md).

## Input Contract

The public API assumes:
- complex-valued input
- shape `(T, H, W)`, where `T` is slow time
- a residual cube that is already clutter-filtered, or beamformed slow-time IQ you intend to compare on the same residual path
- a correct slow-time `prf_hz` for the acquisition

The public API does not assume:
- raw-channel beamforming
- sparse-compounded-frame PD reconstruction
- a replacement clutter filter

If you only have rendered Doppler images or magnitude-only summaries, this is
not the right integration surface.

## When to Use Each Variant

Start with the fixed statistic unless you already know your acquisition is
persistently clutter-dominant. The paper’s deployment rule is intentionally
conservative: the fixed statistic is the default, adaptive is a telemetry-aware
bridge, and the fully whitened variant is a narrower escalation path rather
than a universal upgrade.

- `fixed`: recommended default for first integration and the best public starting point.
- `adaptive`: use when you want the fixed statistic plus guard telemetry and selective whitening on promoted tiles.
- `whitened`: use when representative windows repeatedly show elevated guard-band contamination and your runtime budget can tolerate the extra cost.
- `whitened_power`: use only as an ablation or diagnostic comparison, not as a default deployment rule.

The practical interpretation of the adaptive summary fields is:
- `adaptive_guard_fraction_p90`: high-end guard-band contamination across tiles.
- `adaptive_promote_fraction`: fraction of tiles promoted onto the whitened branch.
- `adaptive_promoted_tiles`: raw count of promoted tiles.

If those telemetry fields stay near zero on representative windows, keep
`fixed`. If they stay elevated and the replay budget remains acceptable, test
`whitened` on the same clutter-filtered residual before changing anything else.

## Minimal Example

```python
import numpy as np

from fus_detectors import DetectorConfig, score_residual_cube

residual_cube = np.load("clutter_filtered_residual.npy")  # complex64, shape (T, H, W)

result = score_residual_cube(
    residual_cube,
    prf_hz=3000.0,
    config=DetectorConfig(
        variant="fixed",
        device="cpu",
    ),
)

readout_map = result.readout_map
score_map = result.score_map
summary = result.summary.to_dict()
```

## Supported Public Variants

- `fixed`: the fixed matched-subspace statistic. This is the default and the
  recommended drop-in starting point.
- `adaptive`: the fixed statistic plus label-free guard telemetry. It keeps the
  fixed statistic by default and promotes clutter-dominant tiles onto the
  whitened branch when guard-band contamination stays high.
- `whitened`: the fully whitened matched-subspace statistic.
- `whitened_power`: total whitened slow-time power, kept as a bounded ablation.

The public API keeps the stable detector surface small. If you need the larger
research script surface used for paper reproduction, keep using the repo’s
existing scripts under `scripts/`.

## Returned Maps

- `readout_map`: a PD-style detector readout suitable for display or for
  swapping into an existing downstream map-rendering step.
- `score_map`: the primary right-tail detector score map used for thresholding,
  ROC evaluation, and acceptance-style comparisons.

## Reproducibility Tips

- Persist `DetectorConfig.to_dict()` alongside any saved score maps.
- Log `DetectorResult.summary.to_dict()` for quick run-to-run comparison.
- For the adaptive variant, persist the `adaptive_*` summary fields so you can
  track when and why tiles were promoted onto the whitened branch.
- Keep the residual cube fixed when comparing variants. That is the central
  same-residual evaluation discipline used throughout the repo and manuscript.
