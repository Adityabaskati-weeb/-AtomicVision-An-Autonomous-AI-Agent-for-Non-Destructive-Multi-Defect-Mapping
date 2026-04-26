## Sample Inputs

These files are lightweight example spectra for the AtomicVision demo UI.

- `sample-medium.csv`
- `sample-medium.txt`
- `sample-medium.json`
- `sample-early-defect.txt`
- `sample-late-defect.json`

Notes:

- `sample-medium.csv`, `sample-medium.txt`, and `sample-medium.json` now carry
  the same 20-point spectrum in three formats, so they should produce the same
  backend result.
- `sample-early-defect.txt` and `sample-late-defect.json` are intentionally
  different spectra so the live demo returns visibly different defect candidates,
  confidence, and explainability panels.
- Uploaded files now drive the Space's backend upload analysis path directly.
