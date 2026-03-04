# Phased Refactor Plan

## Success Criteria
- Public-repo quality structure and documentation.
- Reduced maintenance burden (less duplication, clearer ownership boundaries).
- Improved runtime performance in STAP hot paths.
- Paper claims remain reproducible at every phase boundary.

## Phase Sequence

### Phase 0: Baseline Freeze
Deliverables:
- Clean baseline tag and manifest refresh.
- Quick verification gate finalized and passing.
- Core/experiments/legacy inventory completed.

Exit gate:
- `scripts/verify_refactor.py --mode quick --execute`

### Phase 1: Verification Harness + Governance
Deliverables:
- Stable `quick/phase/full` verification harness.
- Threshold contract and failure policy.
- CI-ready command surface for refactor validation.
- Phase-close report generation template/tooling.

Exit gate:
- `scripts/verify_refactor.py --mode phase --execute`

### Phase 2: Structural Cleanup
Deliverables:
- Move/rename modules into a cleaner package layout.
- De-duplicate orchestration scripts.
- Archive/remove approved legacy candidates.
- Execute per-chunk gate cadence from `docs/refactor/PHASE2_KICKOFF_CHECKLIST.md`.

Exit gate:
- `scripts/verify_refactor.py --mode phase --execute`

### Phase 3: Core Runtime Refactor
Deliverables:
- Clear separation between algorithm, orchestration, and I/O.
- Better debug telemetry and deterministic replay hooks.
- Simplified API boundaries around STAP core.
- Execute per-chunk boundary plan from `docs/refactor/PHASE3_KICKOFF_CHECKLIST.md`.

Exit gate:
- `scripts/verify_refactor.py --mode phase --execute`

### Phase 4: Performance Refactor
Deliverables:
- Profile-driven optimization of top hotspots.
- Better CUDA/Triton launch efficiency where applicable.
- Performance toggles and rollback-safe paths.

Exit gate:
- `scripts/verify_refactor.py --mode phase --execute`
- plus latency sanity from phase gate within agreed drift bounds.

### Phase 5: Public Release Hardening
Deliverables:
- Professional repo polish (docs, onboarding, contribution guide).
- Clean release tag with reproducibility commands.
- Final paper/supplement reproduction package.

Exit gate:
- `scripts/verify_refactor.py --mode full --execute`

## Verification Cadence
- Per major PR/chunk: `quick`
- Per phase close: `phase`
- Per milestone/release: `full`

## Commit Discipline
- Commit at each phase close (required).
- Suggested message style: `phaseN: <outcome>`
- Phase-close commit body must include:
  - verification mode + result
  - key metric drift summary
  - manifest updates
