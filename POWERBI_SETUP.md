# Power BI Setup Guide

This project now ships a **FastAPI data service** that exposes the risk model's
output as REST endpoints. Power BI Desktop consumes those endpoints to render
the dashboards that used to live in the Streamlit app.

> **Note** — The Streamlit app (`src/app.py`) is kept as a backup and still
> runs. The FastAPI service is the new official frontend data layer.

---

## 1. Prerequisites

| Tool | Version | Notes |
| --- | --- | --- |
| Python | 3.10+ | Same env that runs the project |
| Power BI Desktop | latest | Free download from Microsoft |
| Network | localhost | No cloud / Azure AD needed for Desktop |

Install the new Python dependencies **inside the project venv** (this repo
ships a `.venv` folder — make sure you use it, not your system Python):

```powershell
.\.venv\Scripts\Activate.ps1     # activate the venv first
pip install -r requirements.txt
```

This adds `fastapi` and `uvicorn` to the existing requirements.

---

## 2. Start the API

From the project root, **with the venv activated**:

```powershell
.\.venv\Scripts\Activate.ps1
uvicorn src.api.main:app --reload --port 8000
```

If you'd rather not activate the venv, call its Python directly:

```powershell
.\.venv\Scripts\python.exe -m uvicorn src.api.main:app --reload --port 8000
```

> **Common error:** `ModuleNotFoundError: No module named 'sqlalchemy'` means
> uvicorn was launched from your **system Python** instead of the venv.
> Activate the venv (or use the explicit `.venv\Scripts\python.exe -m uvicorn`
> form above) and re-run.

You should see:

```
INFO:     Uvicorn running on http://127.0.0.1:8000
```

Open <http://localhost:8000> in a browser — you'll get a landing page that
lists every endpoint, plus <http://localhost:8000/docs> for full Swagger docs.

### Endpoint summary

| URL | Format | Purpose |
| --- | --- | --- |
| `/csv/companies` | CSV | One row per company-period with risk scores, ratings, and key ratios |
| `/csv/risk-distribution` | CSV | Count + share of companies in each rating bucket (AAA..D) |
| `/csv/industry-benchmarks` | CSV | Mean risk + ratios grouped by industry |
| `/csv/time-series` | CSV | Avg risk per `as_of_date` |
| `/csv/scenarios` | CSV | Stress-test results (Basel adverse / severe, IFRS, etc.) |
| `/csv/var` | CSV | Value-at-Risk matrix (3 methods x 3 confidence levels) |
| `/csv/model-metrics` | CSV | Held-out test metrics per model (ROC-AUC, PR-AUC, Brier, precision/recall/F1) |
| `/model/metrics` | JSON | Full evaluation payload: metrics, ROC/PR/calibration curves, confusion matrix, feature importance |
| `/companies` etc. | JSON | Same data as JSON if you prefer the Web JSON connector |
| `POST /refresh` | JSON | Rebuilds the cached scored dataset |

---

## 3. Connect Power BI Desktop

### 3a. Add the companies fact table

1. Open **Power BI Desktop**.
2. Ribbon: **Home -> Get Data -> Web**.
3. Choose **Basic**, paste:
   ```
   http://localhost:8000/csv/companies
   ```
4. Click **OK**. When prompted for authentication choose **Anonymous** and **Connect**.
5. In the preview window click **Transform Data** (recommended) or **Load**.
6. In Power Query make sure column types are correct:
   - `as_of_date` -> Date
   - `predicted_proba`, `aggregated_risk`, ratio columns -> Decimal Number
   - `risk_rating` -> Text
7. Click **Close & Apply**.

### 3b. Add the supporting tables

Repeat the **Get Data -> Web** flow for each of these URLs, loading each into
its own table:

```
http://localhost:8000/csv/risk-distribution
http://localhost:8000/csv/industry-benchmarks
http://localhost:8000/csv/time-series
http://localhost:8000/csv/scenarios
http://localhost:8000/csv/var
```

### 3c. Build relationships (optional but recommended)

In **Model view**:

- `companies[industry]` -> `industry-benchmarks[industry]` (many-to-one)
- `companies[risk_rating]` -> `risk-distribution[risk_rating]` (many-to-one)
- `companies[as_of_date]` -> `time-series[as_of_date]` (many-to-one)

Power BI usually auto-detects these.

---

## 4. Recommended visuals

A solid replica of the old Streamlit dashboard:

| Page | Visual | Field mapping |
| --- | --- | --- |
| Portfolio Overview | KPI card | Avg `aggregated_risk` from `companies` |
| Portfolio Overview | KPI card | Count of `company_id` |
| Portfolio Overview | KPI card | % of rows where `risk_flag = 1` |
| Portfolio Overview | Donut | Axis: `risk_rating`, Values: `count` (from `risk-distribution`) |
| Industry View | Stacked bar | Axis: `industry`, Value: `aggregated_risk` (mean) |
| Industry View | Table | All `industry-benchmarks` columns |
| Trends | Line chart | X: `as_of_date`, Y: `avg_risk`, `avg_proba` |
| Stress Testing | Bar chart | Axis: `scenario`, Value: `mean_default_proba` |
| Stress Testing | Table | All `scenarios` columns |
| VaR | Matrix | Rows: `method`, Columns: `confidence`, Values: `value_at_risk` |

### Conditional formatting for risk ratings

In the companies table, apply conditional formatting on `risk_rating`:

```
AAA, AA, A          -> green   #00AA00, #44CC44, #88DD88
BBB, BB             -> yellow  #FFDD00, #FFAA00
B, CCC              -> orange  #FF7700, #FF5500
CC, C, D            -> red     #FF3300, #CC0000, #990000
```

This matches the colour scheme used by `src/risk_rating.py`.

---

## 5. Refreshing the data

- **Manual:** In Power BI ribbon click **Home -> Refresh**. The API will
  return the currently cached scored dataset.
- **Rebuild the cache:** Hit `POST http://localhost:8000/refresh` (e.g. via
  the Swagger UI button at `/docs`). This re-runs feature engineering,
  retrains the models, and re-scores all companies.
- **Scheduled refresh:** Configure Power BI's scheduled refresh against the
  same URL. If you publish the report to the Power BI Service you'll need a
  Power BI Gateway pointing at the machine running uvicorn.

---

## 6. Publishing to the Power BI Service (optional)

Localhost works only on your own machine. To share the report:

1. Run the FastAPI service on a host reachable from the internet (or your
   VPN). For example deploy to Azure App Service / Railway / Render and
   replace `http://localhost:8000` with the public URL.
2. In Power BI Desktop, open Power Query and **edit the data source URL** to
   the public one.
3. **Publish** the report (Home ribbon -> Publish).
4. In the Power BI Service, set up **Scheduled Refresh**. For non-public
   endpoints install the **On-Premises Data Gateway** and add the API host as
   an allowed data source.

For embedded scenarios (Power BI inside a web page) you'll need a Power BI
Pro / PPU licence and Azure AD app registration — outside the scope of this
guide.

---

## 7. Troubleshooting

| Symptom | Fix |
| --- | --- |
| Power BI says "Unable to connect" | Make sure `uvicorn` is running and that nothing else is on port 8000 |
| Date column treated as text | In Power Query set the column type to Date |
| Empty `/scenarios` result | Check that `config/scenarios.yaml` is intact |
| `/companies` returns 0 rows | The configured data file in `config/model_config.yaml` may be missing; the API falls back to `data/expanded_real_world_financials_fixed.csv` |
| 500 error on first request | The model is still being trained; wait ~10s then retry — the first call warms the cache |

---

## 8. Going further

- **Direct database source:** Swap the CSV in `config/model_config.yaml` for
  the Postgres block. Power BI will see the same endpoints transparently.
- **Custom KPIs:** Add new endpoints in `src/api/main.py` — they're just
  functions returning a `pd.DataFrame`.
- **Excel users:** Every `/csv/*` URL works in Excel too via **Data -> From
  Web**.
