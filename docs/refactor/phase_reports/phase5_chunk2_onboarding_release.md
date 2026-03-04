# Phase 5 Chunk 2: Onboarding and Release Hardening

## Scope
- Improve external onboarding surface.
- Add release/citation metadata and public CI entrypoint.

## Changes
- Rewrote onboarding README:
  - `README.md`
- Added release checklist:
  - `docs/PUBLIC_RELEASE_CHECKLIST.md`
- Added changelog scaffold:
  - `CHANGELOG.md`
- Added citation metadata:
  - `CITATION.cff`
- Added public CI workflow:
  - `.github/workflows/public_ci.yml`

## Validation
- `make refactor-quick` passed after these changes.

## Notes
- This chunk is documentation/CI metadata hardening only.
- Runtime behavior and detector math were not changed.
