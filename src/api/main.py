"""FastAPI service exposing risk analytics to Power BI (and any other BI tool).

Run locally:
    uvicorn src.api.main:app --reload --port 8000

Then point Power BI Desktop at:
    http://localhost:8000/csv/companies        (Get Data -> Web)
    http://localhost:8000/csv/risk-distribution
    ...

See POWERBI_SETUP.md for end-to-end instructions.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .state import SRC_DIR, build_state, get_state
import sys

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import config  # noqa: E402
from features import engineer_features  # noqa: E402
from scenarios import run_scenarios  # noqa: E402
from var_calculations import (  # noqa: E402
    historical_var,
    monte_carlo_var,
    parametric_var,
)


app = FastAPI(
    title="Enterprise Risk Intelligence API",
    description=(
        "REST data layer for the Risk Prediction platform. "
        "Designed for consumption by Power BI Desktop, Excel, or any BI client "
        "via the Web/JSON or CSV connector."
    ),
    version="1.0.0",
)

# Power BI Desktop talks to localhost; allow all origins for ease of use.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _df_to_csv_response(df: pd.DataFrame, filename: str) -> StreamingResponse:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def _df_to_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    safe = df.copy()
    for col in safe.columns:
        if pd.api.types.is_datetime64_any_dtype(safe[col]):
            safe[col] = safe[col].dt.strftime("%Y-%m-%d")
    safe = safe.replace({np.nan: None, np.inf: None, -np.inf: None})
    return safe.to_dict(orient="records")


def _select_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    present = [c for c in cols if c in df.columns]
    return df[present].copy()


COMPANY_BASE_COLS = [
    "company_id",
    "company_name",
    "symbol",
    "as_of_date",
    "industry",
    "sector",
    "region",
    "revenue",
    "ebitda",
    "debt",
    "cash",
    "total_assets",
    "total_equity",
    "predicted_proba",
    "risk_rating",
    "rating_numeric",
    "leverage_ratio",
    "debt_to_equity",
    "current_ratio",
    "quick_ratio",
    "interest_coverage",
    "ebitda_margin",
    "net_margin",
    "roa",
    "roe",
    "altman_z_score",
    "financial_score",
    "operational_score",
    "compliance_score",
    "aggregated_risk",
    "risk_flag",
]


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@app.on_event("startup")
def _warm_state() -> None:
    build_state(force=False)


# ---------------------------------------------------------------------------
# Dashboard + health
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
def dashboard() -> FileResponse:
    """Serve the single-page dashboard."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> Dict[str, Any]:
    state = get_state()
    return {
        "status": "ok",
        "rows": int(len(state.scored)) if state.scored is not None else 0,
        "last_built": state.last_built.isoformat() if state.last_built is not None else None,
    }


@app.post("/refresh")
def refresh() -> Dict[str, Any]:
    state = build_state(force=True)
    return {
        "status": "rebuilt",
        "rows": int(len(state.scored)) if state.scored is not None else 0,
        "last_built": state.last_built.isoformat() if state.last_built is not None else None,
    }


# ---------------------------------------------------------------------------
# /model — training metrics, curves & feature importance
# ---------------------------------------------------------------------------

@app.get("/model/metrics")
def model_metrics() -> Dict[str, Any]:
    """Evaluation metrics for every trained model (held-out test set)."""
    state = get_state()
    trained = state.trained
    active = "xgboost" if trained.xgb_pipeline is not None else "logistic"
    return {
        "active_model": active,
        "data": trained.data_summary,
        "models": trained.metrics,
        "feature_importance": trained.feature_importance,
    }


@app.get("/csv/model-metrics")
def model_metrics_csv() -> StreamingResponse:
    state = get_state()
    rows = []
    for name, m in state.trained.metrics.items():
        rows.append({
            "model": name,
            "roc_auc": m["roc_auc"],
            "pr_auc": m["pr_auc"],
            "brier_score": m["brier_score"],
            "accuracy": m["accuracy"],
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
        })
    return _df_to_csv_response(pd.DataFrame(rows), "model_metrics.csv")


# ---------------------------------------------------------------------------
# /companies — main fact table for Power BI
# ---------------------------------------------------------------------------

def _companies_frame(
    industry: Optional[str],
    rating: Optional[str],
    min_risk: Optional[float],
    limit: Optional[int],
) -> pd.DataFrame:
    state = get_state()
    df = state.scored.copy()
    if industry:
        df = df[df["industry"].astype(str).str.lower() == industry.lower()]
    if rating:
        df = df[df["risk_rating"].astype(str).str.upper() == rating.upper()]
    if min_risk is not None:
        df = df[df["aggregated_risk"] >= float(min_risk)]
    df = _select_columns(df, COMPANY_BASE_COLS)
    if limit:
        df = df.head(int(limit))
    return df


@app.get("/companies")
def companies(
    industry: Optional[str] = Query(None, description="Filter by industry name"),
    rating: Optional[str] = Query(None, description="Filter by risk rating (AAA..D)"),
    min_risk: Optional[float] = Query(None, ge=0.0, le=1.0),
    limit: Optional[int] = Query(None, gt=0, le=10000),
) -> JSONResponse:
    df = _companies_frame(industry, rating, min_risk, limit)
    return JSONResponse(content=_df_to_records(df))


@app.get("/csv/companies")
def companies_csv(
    industry: Optional[str] = Query(None),
    rating: Optional[str] = Query(None),
    min_risk: Optional[float] = Query(None, ge=0.0, le=1.0),
    limit: Optional[int] = Query(None, gt=0, le=10000),
) -> StreamingResponse:
    df = _companies_frame(industry, rating, min_risk, limit)
    return _df_to_csv_response(df, "companies.csv")


@app.get("/company/{company_id}")
def company_detail(company_id: str) -> JSONResponse:
    state = get_state()
    df = state.scored
    match = df[df["company_id"].astype(str) == company_id]
    if match.empty:
        raise HTTPException(status_code=404, detail=f"Company '{company_id}' not found")
    record = _df_to_records(match)[0]
    return JSONResponse(content=record)


# ---------------------------------------------------------------------------
# /risk-distribution
# ---------------------------------------------------------------------------

RATING_ORDER = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC", "CC", "C", "D"]


def _risk_distribution_frame() -> pd.DataFrame:
    state = get_state()
    df = state.scored
    counts = df["risk_rating"].value_counts().reindex(RATING_ORDER, fill_value=0)
    total = int(counts.sum())
    out = pd.DataFrame({
        "risk_rating": counts.index,
        "count": counts.values,
        "share": (counts.values / total) if total else 0.0,
    })
    return out


@app.get("/risk-distribution")
def risk_distribution() -> JSONResponse:
    return JSONResponse(content=_df_to_records(_risk_distribution_frame()))


@app.get("/csv/risk-distribution")
def risk_distribution_csv() -> StreamingResponse:
    return _df_to_csv_response(_risk_distribution_frame(), "risk_distribution.csv")


# ---------------------------------------------------------------------------
# /industry-benchmarks
# ---------------------------------------------------------------------------

def _industry_benchmarks_frame() -> pd.DataFrame:
    state = get_state()
    df = state.scored
    agg_cols = {
        "predicted_proba": "mean",
        "aggregated_risk": "mean",
        "leverage_ratio": "mean",
        "current_ratio": "mean",
        "ebitda_margin": "mean",
        "altman_z_score": "mean",
        "company_id": "count",
    }
    agg_cols = {k: v for k, v in agg_cols.items() if k in df.columns}
    grouped = df.groupby("industry", dropna=False).agg(agg_cols).reset_index()
    grouped = grouped.rename(columns={"company_id": "company_count"})
    return grouped.sort_values("aggregated_risk", ascending=False)


@app.get("/industry-benchmarks")
def industry_benchmarks() -> JSONResponse:
    return JSONResponse(content=_df_to_records(_industry_benchmarks_frame()))


@app.get("/csv/industry-benchmarks")
def industry_benchmarks_csv() -> StreamingResponse:
    return _df_to_csv_response(_industry_benchmarks_frame(), "industry_benchmarks.csv")


# ---------------------------------------------------------------------------
# /time-series
# ---------------------------------------------------------------------------

def _time_series_frame(industry: Optional[str]) -> pd.DataFrame:
    state = get_state()
    df = state.scored.copy()
    if industry:
        df = df[df["industry"].astype(str).str.lower() == industry.lower()]
    df["as_of_date"] = pd.to_datetime(df["as_of_date"])
    grouped = (
        df.groupby("as_of_date")
        .agg(
            avg_risk=("aggregated_risk", "mean"),
            avg_proba=("predicted_proba", "mean"),
            avg_leverage=("leverage_ratio", "mean"),
            company_count=("company_id", "count"),
        )
        .reset_index()
        .sort_values("as_of_date")
    )
    return grouped


@app.get("/time-series")
def time_series(industry: Optional[str] = Query(None)) -> JSONResponse:
    return JSONResponse(content=_df_to_records(_time_series_frame(industry)))


@app.get("/csv/time-series")
def time_series_csv(industry: Optional[str] = Query(None)) -> StreamingResponse:
    return _df_to_csv_response(_time_series_frame(industry), "time_series.csv")


# ---------------------------------------------------------------------------
# /scenarios
# ---------------------------------------------------------------------------

def _scenarios_frame() -> pd.DataFrame:
    state = get_state()
    scenarios_cfg = config.load_scenarios()
    feature_cols = state.trained.feature_cols

    def infer(shocked_df: pd.DataFrame) -> np.ndarray:
        shocked_feat = engineer_features(shocked_df)
        return state.pipeline.predict_proba(shocked_feat[feature_cols])[:, 1]

    # Pass the full scored frame so engineer_features has every raw column it needs
    results = run_scenarios(state.scored, scenarios_cfg, infer)

    rows = []
    for name, payload in results.items():
        proba = np.asarray(payload["probability"])
        rows.append({
            "scenario": name,
            "description": payload.get("description", ""),
            "mean_default_proba": float(proba.mean()),
            "median_default_proba": float(np.median(proba)),
            "max_default_proba": float(proba.max()),
            "p95_default_proba": float(np.percentile(proba, 95)),
            "companies_at_risk": int((proba >= 0.20).sum()),
            "company_count": int(len(proba)),
        })
    return pd.DataFrame(rows).sort_values("mean_default_proba", ascending=False)


@app.get("/scenarios")
def scenarios() -> JSONResponse:
    return JSONResponse(content=_df_to_records(_scenarios_frame()))


@app.get("/csv/scenarios")
def scenarios_csv() -> StreamingResponse:
    return _df_to_csv_response(_scenarios_frame(), "scenarios.csv")


# ---------------------------------------------------------------------------
# /var — Value at Risk
# ---------------------------------------------------------------------------

def _portfolio_returns() -> pd.Series:
    """Synthetic portfolio returns derived from predicted default probabilities.

    We don't have a real return series here, so we project a notional daily P&L
    derived from the inverse of predicted_proba. This is sufficient for the
    BI surface and matches what the Streamlit app was charting.
    """
    state = get_state()
    proba = state.scored["predicted_proba"].to_numpy()
    rng = np.random.default_rng(seed=42)
    base = 0.0005 - proba * 0.001
    sigma = 0.01 + proba * 0.02
    returns = rng.normal(loc=base, scale=sigma, size=(252, len(proba)))
    return pd.Series(returns.mean(axis=1))


@app.get("/var")
def var_endpoint(
    method: str = Query("historical", regex="^(historical|parametric|monte_carlo)$"),
    confidence: float = Query(0.95, gt=0.5, lt=1.0),
) -> Dict[str, Any]:
    returns = _portfolio_returns()
    if method == "historical":
        value = float(historical_var(returns, confidence_level=confidence))
    elif method == "parametric":
        value = float(parametric_var(returns, confidence_level=confidence))
    else:
        value = float(monte_carlo_var(returns, confidence_level=confidence))
    return {
        "method": method,
        "confidence": confidence,
        "value_at_risk": value,
        "mean_return": float(returns.mean()),
        "std_return": float(returns.std()),
        "observations": int(len(returns)),
    }


# ---------------------------------------------------------------------------
# /correlations — Pearson correlations between key financial ratios
# ---------------------------------------------------------------------------

CORR_FIELDS = [
    "predicted_proba",
    "leverage_ratio",
    "debt_to_equity",
    "current_ratio",
    "quick_ratio",
    "interest_coverage",
    "ebitda_margin",
    "net_margin",
    "roa",
    "roe",
    "altman_z_score",
    "aggregated_risk",
]


@app.get("/correlations")
def correlations() -> Dict[str, Any]:
    state = get_state()
    cols = [c for c in CORR_FIELDS if c in state.scored.columns]
    df = state.scored[cols].replace([np.inf, -np.inf], np.nan).dropna()
    corr = df.corr().round(3)
    return {
        "fields": cols,
        "matrix": corr.values.tolist(),
    }


# ---------------------------------------------------------------------------
# /returns — raw portfolio returns series feeding the VaR calc
# ---------------------------------------------------------------------------

@app.get("/returns")
def returns_series() -> Dict[str, Any]:
    r = _portfolio_returns()
    return {
        "returns": r.tolist(),
        "mean": float(r.mean()),
        "std": float(r.std()),
        "observations": int(len(r)),
        "var_95_historical": float(historical_var(r, confidence_level=0.95)),
    }


@app.get("/csv/var")
def var_csv() -> StreamingResponse:
    returns = _portfolio_returns()
    rows = []
    for conf in [0.90, 0.95, 0.99]:
        for method, fn in [
            ("historical", historical_var),
            ("parametric", parametric_var),
            ("monte_carlo", monte_carlo_var),
        ]:
            rows.append({
                "method": method,
                "confidence": conf,
                "value_at_risk": float(fn(returns, confidence_level=conf)),
            })
    df = pd.DataFrame(rows)
    return _df_to_csv_response(df, "var.csv")
