# Public Release Checklist

Use this checklist before creating a public release tag.

## 1) Repository Hygiene
- [ ] `git status` is clean.
- [ ] No private paths, tokens, or credentials in tracked files.
- [ ] Large generated outputs are excluded from git.

## 2) Documentation
- [ ] `README.md` reflects current setup and reproduction entry points.
- [ ] `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md` are present.
- [ ] Citation metadata (`CITATION.cff`) is present and accurate.
- [ ] Release notes/changelog updated.

## 3) Reproducibility
- [ ] `repro_manifest.json` is refreshed and consistent with scripts.
- [ ] At least one smoke reproducibility command is validated.
- [ ] Paper-critical commands in docs map to existing scripts/targets.

## 4) Validation Gates
- [ ] `make refactor-quick` passes.
- [ ] `make refactor-phase` passes.
- [ ] `make refactor-full` passes (or documented exception accepted).

## 5) Release Artifact Prep
- [ ] Version tag selected (e.g., `v0.1.0`).
- [ ] Changelog section for the tag is finalized.
- [ ] Release notes include:
  - core changes,
  - validation commands,
  - known limitations.
