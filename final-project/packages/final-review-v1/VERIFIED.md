# Review archive verification

`24.zip` is a DRAFT review package for Group 24, not a submission receipt.

- Archive: 2,913,268 bytes, 225 entries; all 224 indexed hashes and ZIP CRCs passed.
- SHA-256: `7e1d86708471b0d70a4be375ff7211f69a12505802d626e951f2a0fdbc4fd0bb`.
- Extracted-code suite: 216 discovered, 211 passed, five PDF checks deferred to
  the rendering environment. All 23 extracted report tests passed there,
  including those five. Zero failures.
- Seven report pages were visually inspected; Times New Roman 12pt, 1.15 spacing.
  Main discussions have 94, 122 and 113 words.
- All executable source and tests exactly match the archived copies. A later
  17-line editorial clearance was appended to repository AUDIT.md; the archive
  preserves the audit snapshot taken at packaging. The report PDF is unchanged.

The archive puts `report.pdf`, `evidence/` and `demo/` at its root and executable
source under `code/`. Repository links in `code/README.md` describe the full
checkout layout; use these root locations for packaged review artifacts.
The main reproduction commands are run from `code/` with the documented data
and instructor fork. Tests here used already-verified local macOS runtimes;
a fresh network installation or another operating system was not tested.

`verification.json` records checksums and exact test scope. No push or submission
was performed. Names, actual contributions and real individual peer feedback
remain human inputs.
