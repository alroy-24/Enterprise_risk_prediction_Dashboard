# Enterprise Risk Intelligence Platform (ERIP) – Consulting-Style MVP

**Big-4 / Enterprise aligned risk analytics platform** with **SQL-first data analysis**, market risk metrics (VaR, PnL attribution), and BI-style dashboards. 

This repo demonstrates **SQL skills**, **time-series analysis**, **market data processing** (equities, fixed income, spreads), and **value-at-risk calculations** alongside traditional enterprise risk features. Built for data analysts working in finance/risk analytics.

## Quickstart

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run src/app.py
```

## What’s included

### SQL & Data Analysis (Industry-Standard)
- **SQL-first architecture**: PostgreSQL integration with industry-standard queries
- **SQL views and stored procedures** for risk analytics (aggregations, window functions, CTEs)
- **SQL queries module** (`sql/risk_analytics_queries.sql`) demonstrating advanced SQL skills
- **Data export to Excel/CSV** for BI tools (Tableau/PowerBI-ready)

### Market Risk & Time-Series Analysis
- **VaR calculations**: Parametric, Historical, and Monte Carlo VaR methods
- **PnL Attribution**: Portfolio performance breakdown by risk factors
- **Market data simulation**: Equity prices, fixed income yields, credit spreads, VIX-style volatility
- **Time-series analysis**: Rolling metrics, returns, cumulative performance

### Enterprise Risk Features
- **Data layer**: CSV/Excel/Postgres loaders with validation
- **Feature engineering**: Industry-standard financial ratios (Altman Z-score, leverage, liquidity, coverage ratios)
- **Models**: Logistic Regression + XGBoost with scikit-learn pipelines
- **Explainability**: SHAP global/local explanations
- **Scenario engine**: Regulatory stress scenarios (Basel III, IFRS 9) + custom business scenarios
- **Risk ratings**: Credit-style ratings (AAA to D)
- **Peer benchmarking**: Industry/region comparisons
- **Governance**: MLflow, Great Expectations, Evidently hooks (optional)

## Repo layout

```
config/
  weights.yaml          # risk aggregation weights
  scenarios.yaml        # stress/scenario definitions
  model_config.yaml     # model hyperparameters + flags
data/
  sample_financials.csv # demo dataset
src/
  app.py                # Streamlit BI dashboard (8 views)
  config.py             # config loading helpers
  data_ingest.py        # SQL/CSV/Excel loaders
  features.py           # financial ratio engineering
  models.py             # ML training/inference
  explainability.py     # SHAP integration
  scenarios.py          # stress testing engine
  aggregation.py        # risk aggregation
  recommendations.py    # remediation suggestions
  market_data.py        # market data simulation (equities, yields, spreads)
  var_calculations.py   # VaR methods (parametric, historical, Monte Carlo)
  pnl_attribution.py    # PnL attribution analysis
  sql_queries.py        # SQL query execution module
  risk_rating.py        # credit rating system
  benchmarking.py       # peer comparison
  governance/           # MLflow/GE/Evidently hooks
sql/
  risk_analytics_queries.sql  # Industry-standard SQL queries
```

## How to demo

1) Edit or replace `data/sample_financials.csv` with client data (or configure Postgres in `config/model_config.yaml`).
2) Run `streamlit run src/app.py` and explore:
   - **Portfolio Overview**: Risk scorecards, heatmaps, risk ratings
   - **Market Risk**: VaR analysis, PnL attribution, equity/yield trends
   - **SQL Queries**: View industry-standard SQL queries for risk analytics
   - **Peer Benchmarking**: Industry comparisons with percentile rankings
   - **Explainability**: SHAP feature importance
   - **Stress Testing**: Regulatory scenarios (Basel III, IFRS 9)
   - **Recommendations**: Actionable remediation steps
   - **Export**: Excel/CSV reports for BI tools

### Using SQL Queries

The `sql/risk_analytics_queries.sql` file contains production-ready SQL queries demonstrating:
- Aggregations (AVG, PERCENTILE_CONT, COUNT, SUM)
- Window functions (LAG, ROW_NUMBER, PARTITION BY)
- CTEs and subqueries
- JOINs for multi-table analysis

Execute via Python:
```python
from src.sql_queries import get_high_risk_companies, get_industry_benchmarks
from sqlalchemy import create_engine

engine = create_engine("postgresql://user:pass@host:5432/db")
df = get_high_risk_companies(engine)
```

## Governance notes

- Deep learning is intentionally omitted: transparency and regulatory defensibility take priority.
- MLflow hooks are present but off by default; enable and set a tracking URI when needed.
- Great Expectations/Evidently hooks are stubbed for fast PoC; wire to production checks/monitoring for go-live.

## Skills Demonstrated

**SQL & Data Analysis:**
- Complex SQL queries (aggregations, window functions, CTEs, subqueries)
- PostgreSQL integration with SQLAlchemy
- Data export for BI tools (Excel, CSV)

**Market Risk Analytics:**
- VaR calculations (Parametric, Historical, Monte Carlo)
- PnL attribution analysis
- Time-series analysis (equities, fixed income, spreads)
- Risk factor modeling (VIX-style volatility)

**Python & ML:**
- pandas, numpy for data manipulation
- scikit-learn pipelines
- XGBoost for risk prediction
- SHAP for explainability

**BI & Visualization:**
- Streamlit dashboards (8 interactive views)
- Plotly visualizations (charts, heatmaps, time-series)
- Excel export for Tableau/PowerBI integration

## Next steps

- Connect to real market data feeds (Bloomberg, Refinitiv APIs)
- Integrate with existing SQL databases (PostgreSQL, SQL Server)
- Add Tableau/PowerBI connectors
- Deploy to cloud (AWS, Azure, GCP) or Railway/Streamlit Cloud
- Add CI/CD (GitHub Actions) for automated testing


