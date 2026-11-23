# AI Drilling Intelligence — Professional GitHub Test Build

A dark, interactive Streamlit workspace for drilling engineering analytics, machine-learning experiments, time-series baselines and data-driven scenario exploration.

## Included demonstration dataset

- File: `data/drilling_demonstration_dataset.csv.gz`
- Shape: **59,163 observations × 118 columns**
- Compressed size: approximately **26.1 MB** (the uncompressed CSV is approximately 103.9 MB).
- The dataset is generated for demonstration, software testing and analytics development. It is **not field measurements** and must not be represented as actual rig telemetry.
- Column labels have been normalized for presentation. The original engineering values and feature structure are retained.

The compressed CSV is read directly by pandas. Keeping it compressed makes the repository easier to upload and avoids GitHub's 100 MB single-file limit.

## Repository and deployment

- GitHub repository: `Ali-sobhy1/ai-drilling-intelligence`
- Branch: `main`
- Streamlit entry point: `app.py`
- Default dataset: `data/drilling_demonstration_dataset.csv.gz`

Upload the **contents** of this project folder to the repository root. Preserve the `data` folder and its compressed CSV file. In Streamlit Community Cloud, set the main file path to `app.py`.

## Run locally

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Dashboard modules

- Executive Overview — operational KPIs, ROP trend, data-quality indicators, lithology distribution and screening-score distribution.
- Data Explorer — filters, parameter selection, statistical profile, preview and CSV export.
- ROP Optimization — chronological train/test evaluation, model comparison, feature importance and constrained scenario exploration.
- Dysfunction Intelligence — pre-computed screening indicators plus optional Isolation Forest analysis.
- Predictive Analytics — lagged one-step forecasting baseline with chronological evaluation.
- Digital Twin Simulator — data-driven ROP scenario demonstrator using the trained model.
- Advanced Data Science — correlation, PCA and K-Means operating-pattern exploration across numeric fields.
- AI Drilling Assistant — local rule-based drilling guidance and dataset summaries; no paid API or Ollama required.
- Reports & Export — statistical and data-quality tables with Excel export.

## Visual design

The interface uses a near-black / deep-navy operations theme, orange action accents, cyan analytical highlights, dark KPI cards, consistent chart styling, and interactive Plotly charts with hover and zoom controls.

## Important engineering limitations

- The built-in assistant is a deterministic rules layer, not a general-purpose LLM.
- Dysfunction scores and flags are demonstration screening indicators, not confirmed downhole diagnoses or validated event labels.
- The ROP and time-series models are baseline statistical models. Review chronological validation, leakage risk, sampling intervals and operating-mode effects before interpreting performance.
- The Digital Twin module is a data-driven ROP surrogate, not a physics-calibrated operational twin.
- Scenario outputs are not operating instructions or safe setpoints. Apply approved well-specific limits, procedures and engineering authorization.
- The dataset is generated for demonstration and software testing, not field measurements.

## Validation status

The data transformation was checked for **59,163 rows and 118 columns**; the compressed file can be read by pandas. Python syntax compilation and ZIP integrity checks are performed before delivery. A full browser-based Streamlit deployment test still needs to be completed in Streamlit Community Cloud.
