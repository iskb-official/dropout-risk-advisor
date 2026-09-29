# Capacity-Aware Decision Support System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://dropoutra.streamlit.app/)

A light, interpretable Streamlit decision-support workspace for student-retention teams. The current application keeps the existing assessment and cohort-scoring workflow while adding a production-oriented presentation layer, safer PDF layout, responsive containment, first-run orientation, and session-only advisor workspace features.

## Scope and freeze policy

The following application logic is frozen in this release:

- `load_and_train_model`
- `assign_risk_band`
- `get_intervention`
- `get_intervention_detail`
- `LOW_TH`, `HIGH_TH`, `CAPACITY_LIMIT`
- batch allocation order and overflow behavior
- counterfactual calculation
- existing lever/attribution computation
- CSV feature schema
- existing session-state keys and their meanings

The UI/PDF layer was changed without changing those computations.

## Repository structure

The frozen structure is:

```text
app.py
students_dropout_academic_success.csv
requirements.txt
.streamlit/config.toml
README.md
```

No extra runtime service, database, or API is required.

## Local setup

Recommended interpreter for the pinned stack: Python 3.13.

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

The app expects `students_dropout_academic_success.csv` in the same directory as `app.py`.

## Streamlit Community Cloud

1. Push the release to a feature branch.
2. Open a pull request into `main`.
3. In Streamlit Community Cloud, select the repository and `app.py` as the entrypoint.
4. Use Python 3.13 for the deployment environment.
5. Let Community Cloud install `requirements.txt`.
6. Confirm the staging smoke checklist below before promoting the branch.

The current pinned package versions were checked against PyPI on 2026-09-30. Streamlit 1.64.0, XGBoost 3.4.1, SHAP 0.52.0, Plotly 7.1.0, scikit-learn 1.9.1, pandas 3.0.6, NumPy 2.5.3, SciPy 1.18.1, ReportLab 5.0.1, and Kaleido 1.4.0 were selected as the release pins.

### Kaleido / PDF note

Plotly static image export uses Kaleido. Current Kaleido requires Chrome/Chromium. The app deliberately treats chart rendering as optional: if Kaleido/Chrome is unavailable, the PDF still generates with text and tables rather than exposing a stack trace.

## White-label customization

All high-level branding is near the top of `app.py`.

Change:

```python
PRODUCT_NAME = "Capacity-Aware Decision Support System"
INSTITUTION_NAME = "Your Institution"
SUPPORT_URL = ""
MODEL_CARD_URL = ""
LANGUAGE = "en"
```

Use the `COLORS` dictionary for the design tokens:

```python
COLORS = {
    "brand": "#3B5BDB",
    "navy": "#1B2A5C",
    "ink": "#0E1B3D",
    "background": "#F4F6FB",
    "surface": "#FFFFFF",
    "low": "#12A594",
    "medium": "#F5A524",
    "high": "#DC3F4A",
}
```

The current SVG mark is defined in `LOGO_SVG`. Replace that SVG with an institution-safe single-stroke mark when branding it for production.

### Language

UI text introduced by the product layer is grouped in the `UI` dictionary:

```python
UI = {
    "en": {...}
}
```

Add another language key beside `en` and change `LANGUAGE` to activate it. Keep the existing model/business strings unchanged unless they are also translated deliberately as a separate controlled release.

## Privacy posture

The UI explicitly communicates that:

- data stays in the current browser session;
- student PII is not intentionally persisted by this application;
- advisor notes are session-only;
- the application does not require a database;
- no third-party tracking script is intentionally loaded.

Operational deployment should still be reviewed against the institution's own FERPA/GDPR, security, retention, and access-control requirements.

## Accessibility

The release includes:

- visible keyboard focus rings;
- non-color textual risk labels;
- screen-reader helper descriptions for analytical charts;
- responsive wrapping and minimum-width containment;
- reduced-motion handling;
- high-contrast primary text;
- wrapped PDF table cells and ASCII-safe report glyphs.

## Regression checklist

Run these checks before every merge:

| Area | Check | Expected |
|---|---|---|
| Compile | `python -m py_compile app.py` | PASS |
| Startup | `streamlit run app.py` | No startup exception |
| Assessment | Enter profile and click assessment | Probability, band, SHAP, what-if and PDF appear |
| PDF | Download single-student report | PDF opens; no clipped tables |
| Rerun | Change a widget after a generated report | Assessment remains until explicitly replaced |
| What-if | Save 1-5 scenarios | Board persists for current assessment |
| What-if | Save >5 scenarios | Only most recent five are retained |
| Batch upload | Valid CSV | Scoring completes and queue appears |
| Batch upload | Empty CSV | Friendly warning, no stack trace |
| Batch upload | Invalid numeric cells | Coercion/imputation behavior remains unchanged |
| Batch PDF | Download batch PDF | PDF opens and long action text wraps |
| Fairness | Open Fairness Audit | Existing displayed values/claims unchanged |
| Narrow viewport | 320-480 px | No horizontal text overflow |
| Wide viewport | 1920 px | Content remains centered and bounded |
| Accessibility | Keyboard tabbing | Focus is visible |
| Reduced motion | `prefers-reduced-motion` | Animations are disabled |
| PDF fallback | Kaleido unavailable | Text/tables report still generates |

## Deployment staging checklist

Before production promotion:

1. Confirm the CSV in the branch is the intended data file.
2. Confirm `requirements.txt` matches the release.
3. Confirm the institution name, logo, support URL, and model-card URL.
4. Run the full regression checklist.
5. Test Chrome, Edge, Safari, and Firefox.
6. Test a narrow mobile/tablet viewport.
7. Test a full single-student PDF and a batch PDF.
8. Review the privacy notice and institution-specific retention/security policy.
9. Merge through a pull request rather than pushing directly to `main`.

## Rollback

Because this release is intentionally isolated to a feature branch, the simplest rollback is Git-based:

```bash
git switch main
git pull
git revert <merge-commit-sha>
git push origin main
```

For an unmerged branch, close the pull request and redeploy the last known-good commit.

## Custom CI workflow

Your freeze list permits only `.streamlit/config.toml` and `README.md` as new files, so this release does **not** add a `.github/workflows/ci.yml` file. The following is the copy-ready workflow to place in `.github/workflows/ci.yml` once the repository freeze is relaxed:

```yaml
name: CI

on:
  push:
    branches: ["**"]
  pull_request:

jobs:
  quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.13"

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      - name: Compile
        run: python -m py_compile app.py

      - name: AST smoke checks
        run: |
          python - <<'PY'
          import ast
          from pathlib import Path

          source = Path("app.py").read_text(encoding="utf-8")
          tree = ast.parse(source)

          required = {
              "load_and_train_model",
              "assign_risk_band",
              "get_intervention",
              "get_intervention_detail",
              "compute_counterfactual",
              "run_batch",
              "build_pdf_report",
              "build_batch_pdf_report",
          }
          found = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
          missing = required - found
          assert not missing, f"Missing functions: {missing}"
          print("AST_SMOKE_OK")
          PY
```

## Changelog

### 2026-09-30 - UI/PDF production hardening

- Added centralized white-label and language configuration.
- Removed remote font import to keep the runtime more privacy-contained.
- Hardened responsive overflow containment and keyboard focus styling.
- Added a first-run welcome panel and guided tour.
- Added an explicit privacy/session notice.
- Added an advisor closing panel and “assess another student” flow.
- Added a capacity-use progress bar to cohort scoring.
- Improved analytical chart margins, legends, labels, and responsive behavior.
- Added chart accessibility descriptions.
- Rebuilt single-student and batch PDF layouts around wrapped `Paragraph` cells.
- Added true-aspect-ratio chart placement.
- Added branded PDF banner and running header/footer.
- Added `Page x of y` numbering through a numbered ReportLab canvas.
- Added ASCII-safe PDF text normalization.
- Kept Kaleido failure graceful.
- Added empty-upload and batch-scoring error handling without exposing stack traces.
- Kept the frozen model/business logic and existing workflow intact.

## Recommended next step

The next controlled release should be an evidence-backed usability test with advisors/registrars before any deeper workflow or backend changes. Phase 2 items such as multi-tenancy, SSO, roles, audit logs, or a separate model service should remain out of this release.


