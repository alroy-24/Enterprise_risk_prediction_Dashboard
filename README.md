# Enterprise Risk Intelligence Platform (ERIP) – Consulting-Style MVP

This repo is a runnable scaffold for a KPMG-style Enterprise Risk Intelligence Platform. It ingests real-world formats (Excel/CSV/Postgres), engineers risk features, trains explainable models, runs scenario/stress tests, aggregates multi-dimensional risk, and serves CXO-ready visuals in Streamlit.

## Quickstart

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run src/app.py
```

## What’s included

- Data layer: CSV/Excel/Postgres loaders with basic validation hooks.
- Feature engineering: volatility, leverage, liquidity, growth, compliance normalization, industry encoding.
- Models: Logistic Regression (audit-friendly) + XGBoost; pipelines with scaling/encoding.
- Explainability: SHAP summary + per-record explanations.
- Scenario engine: configurable shocks (revenue collapse, debt shock, compliance breach, downturn).
- Aggregation: weighted risk score across financial/operational/compliance axes.
- Recommendations: rule/threshold + SHAP-driven insights.
- Governance stubs: MLflow tracking, Great Expectations template, Evidently drift hook (toggle in config).

## Repo layout

```
config/
  weights.yaml          # risk aggregation weights
  scenarios.yaml        # stress/scenario definitions
  model_config.yaml     # model hyperparameters + flags
data/
  sample_financials.csv # demo dataset
src/
  app.py                # Streamlit UI
  config.py             # config loading helpers
  data_ingest.py        # loaders + validation hooks
  features.py           # feature engineering
  models.py             # training/inference pipelines
  explainability.py     # SHAP integration
  scenarios.py          # scenario/stress engine
  aggregation.py        # risk aggregation
  recommendations.py    # remediation suggestions
  governance/           # stubs for MLflow/GE/Evidently
```

## How to demo

1) Edit or replace `data/sample_financials.csv` with client-like data (or point to Postgres in `config/model_config.yaml`).
2) Adjust weights/scenarios in `config/*.yaml` to match your risk taxonomy.
3) Run `streamlit run src/app.py` and explore:
   - Portfolio view with risk scores and feature drivers.
   - SHAP explanations for transparency.
   - Scenario simulator to see impact of shocks.
   - Recommendations tied to risk signals.

## Governance notes

- Deep learning is intentionally omitted: transparency and regulatory defensibility take priority.
- MLflow hooks are present but off by default; enable and set a tracking URI when needed.
- Great Expectations/Evidently hooks are stubbed for fast PoC; wire to production checks/monitoring for go-live.

## Next steps

- Swap sample data with client feeds and map columns.
- Add authentication/authorization if exposed externally.
- Dockerize (`docker build -t erip .`) or deploy to Streamlit Cloud/AWS EC2.
- Add CI (GitHub Actions) for lint/test/model reproducibility.


