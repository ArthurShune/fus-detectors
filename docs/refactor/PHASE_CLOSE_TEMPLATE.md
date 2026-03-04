# Phase Close Template

Use this template (or the generator script) at the end of every phase so closeout data is consistent.

## Preferred path
```bash
PYTHONPATH=. python scripts/phase_close_report.py --phase <N> --verification-mode phase
```

This writes `docs/refactor/phase_reports/phase<N>_close.md`.

## Required Sections
- Summary
- Verification (exact command + result)
- Metrics and drift vs baseline
- Reproducibility/manifest updates
- Risks
- Handoff to next phase
- Commit record

## Required Verification By Phase
- Phase 0 close: `quick`
- Phase 1-4 close: `phase`
- Phase 5/release close: `full`

## Commit Rule
At least one explicit phase-close commit is required, with message format:
- `phaseN: <outcome>`

Include verification mode/result and key drift notes in the commit body.
