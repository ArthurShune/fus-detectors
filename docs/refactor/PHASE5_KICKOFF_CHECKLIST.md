# Phase 5 Kickoff Checklist (Public Release Hardening)

## Objective
Prepare the repository for external/public use with clear onboarding, governance, and reproducible entrypoints.

## Scope For Phase 5
- Publish-ready docs (`README`, contribution/security/citation policy, release checklist).
- Stable, data-safe CI path for external contributors.
- Clear "smoke" and "paper-critical" reproduction commands.

## Non-Goals
- No detector-math changes.
- No publication-number retuning.
- No large runtime optimization work (already deferred).

## Chunking Plan
1. **Public scaffold chunk**
   - Add phase kickoff notes and OSS governance files.
   - Keep existing code/runtime behavior unchanged.
2. **Onboarding and release-doc chunk**
   - Rewrite `README` for external users.
   - Add release-readiness checklist and citation metadata.
   - Add CI workflow for public-facing checks.
3. **Phase close chunk**
   - Run `make refactor-full` (or agreed release gate).
   - Record closeout with:
     - `make phase-close-report PHASE=5 VERIFY_MODE=full`

## Verification Cadence (Minimal Sweep)
- **Every docs/infra chunk:** `make refactor-quick`
- **Before release tag:** `make refactor-phase`
- **Phase close:** `make refactor-full` (required)

## Acceptance Criteria
- Public onboarding and governance docs are present and coherent.
- Repro commands in docs map to real scripts/targets.
- CI passes on data-safe gates for external PRs.
