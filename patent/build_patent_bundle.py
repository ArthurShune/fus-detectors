#!/usr/bin/env python3
"""Build a filing-style provisional patent package."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PATENT = ROOT / "patent"
GENERATED = PATENT / "generated"

OMIT_HEADINGS = {
    "Inventor / Applicant Placeholders",
    "Open Items For Counsel / Inventor Review",
    "Repo Sources Supporting This Draft",
    "Filing and Scope Notes",
}

RENAME_HEADINGS = {
    "Suggested Claim Families For Later Nonprovisional Drafting": "Claim Appendix",
    "Example Support For Later Claims": "Additional Claim Support",
}


def clean_spec_markdown() -> Path:
    src = ROOT / "provisional_patent_draft_specification.md"
    dst = GENERATED / "provisional_filing_specification.md"
    GENERATED.mkdir(parents=True, exist_ok=True)

    lines = src.read_text().splitlines()
    out: list[str] = []
    skip = False
    skip_alt_titles = False

    for line in lines:
        if line.startswith("# "):
            continue
        if line.startswith("## "):
            heading = line[3:].strip()
            skip = heading in OMIT_HEADINGS
            if skip:
                continue
            heading = RENAME_HEADINGS.get(heading, heading)
            out.append(f"## {heading}")
            skip_alt_titles = False
            continue
        if skip:
            continue
        if line.strip() == "Alternative titles:":
            skip_alt_titles = True
            continue
        if skip_alt_titles:
            if line.startswith("- "):
                continue
            if not line.strip():
                continue
            skip_alt_titles = False
        if not out:
            if "not legal advice" in line.lower():
                continue
            if "organized around invention families" in line.lower():
                continue
            if not line.strip():
                continue
        out.append(line)

    dst.write_text("\n".join(out).strip() + "\n")
    return dst


def run(*cmd: str, cwd: Path | None = None):
    subprocess.run(cmd, cwd=cwd, check=True)


def main():
    clean_md = clean_spec_markdown()
    run(sys.executable, str(PATENT / "generate_patent_figures.py"))
    run(
        "pandoc",
        str(clean_md),
        "-f",
        "gfm",
        "-t",
        "latex",
        "--shift-heading-level-by=-1",
        "--wrap=none",
        "-o",
        str(GENERATED / "provisional_spec_body.tex"),
    )
    run("latexmk", "-pdf", "-interaction=nonstopmode", "provisional_application_package.tex", cwd=PATENT)
    run("latexmk", "-pdf", "-interaction=nonstopmode", "provisional_drawings.tex", cwd=PATENT)
    run("latexmk", "-pdf", "-interaction=nonstopmode", "provisional_submission_checklist.tex", cwd=PATENT)
    run("latexmk", "-pdf", "-interaction=nonstopmode", "provisional_cover_information.tex", cwd=PATENT)


if __name__ == "__main__":
    main()
