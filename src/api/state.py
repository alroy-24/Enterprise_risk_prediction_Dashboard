"""Shared in-memory state for the API.

The dataset and trained models are expensive to build, so we materialise
them once at startup and reuse them across requests. Power BI typically
hits these endpoints with periodic refreshes (every 15-60 min), which
makes the in-memory cache more than adequate.
"""
from __future__ import annotations

import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import config  # noqa: E402
from aggregation import aggregate_scores  # noqa: E402
from data_ingest import DataSource, load_data  # noqa: E402
from features import engineer_features  # noqa: E402
from models import get_inference_pipeline, train_models  # noqa: E402
from risk_rating import add_risk_ratings  # noqa: E402


@dataclass
class AppState:
    cfg: Dict[str, Any] = field(default_factory=dict)
    scored: Optional[pd.DataFrame] = None
    trained: Optional[Any] = None
    pipeline: Optional[Any] = None
    weights: Dict[str, float] = field(default_factory=dict)
    last_built: Optional[pd.Timestamp] = None


_state = AppState()
_lock = threading.Lock()


def _normalise_weights(raw: Dict[str, Any]) -> Dict[str, float]:
    if "financial_weight" in raw:
        return {
            "financial": float(raw["financial_weight"]),
            "operational": float(raw["operational_weight"]),
            "compliance": float(raw["compliance_weight"]),
        }
    return {k: float(v) for k, v in raw.items() if k in {"financial", "operational", "compliance"}}


def build_state(force: bool = False) -> AppState:
    """Build (or rebuild) the scored dataset and cache it."""
    with _lock:
        if _state.scored is not None and not force:
            return _state

        cfg = config.load_model_config()
        data_path = SRC_DIR.parent / cfg["data"]["source"]
        if not data_path.exists():
            # Fallback to the expanded real-world dataset if the configured one is missing
            fallback = SRC_DIR.parent / "data" / "expanded_real_world_financials_fixed.csv"
            data_path = fallback if fallback.exists() else data_path

        df = load_data(DataSource(path=data_path))
        trained = train_models(df, cfg)
        pipeline = get_inference_pipeline(trained, prefer_xgb=True)

        df_feat = engineer_features(df)
        df_feat["predicted_proba"] = pipeline.predict_proba(df_feat[trained.feature_cols])[:, 1]
        df_feat = add_risk_ratings(df_feat)

        weights_raw = config.load_weights()
        weights = _normalise_weights(weights_raw)
        df_scored = aggregate_scores(df_feat, weights)

        _state.cfg = cfg
        _state.scored = df_scored
        _state.trained = trained
        _state.pipeline = pipeline
        _state.weights = weights
        _state.last_built = pd.Timestamp.utcnow()
        return _state


def get_state() -> AppState:
    if _state.scored is None:
        return build_state()
    return _state
