import logging
from dataclasses import dataclass
from typing import Dict, Tuple, List

import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - optional
    XGBClassifier = None  # type: ignore

from features import engineer_features, build_preprocessor

logger = logging.getLogger(__name__)


@dataclass
class TrainedModels:
    logistic_pipeline: Pipeline
    xgb_pipeline: Pipeline | None
    reports: Dict[str, str]
    feature_cols: List[str]


def build_logistic(logistic_cfg: Dict) -> LogisticRegression:
    return LogisticRegression(
        C=logistic_cfg.get("C", 1.0),
        penalty=logistic_cfg.get("penalty", "l2"),
        solver="lbfgs",
        max_iter=logistic_cfg.get("max_iter", 500),
    )


def build_xgb(xgb_cfg: Dict) -> XGBClassifier:
    if XGBClassifier is None:
        raise ImportError("xgboost is not installed.")
    return XGBClassifier(
        max_depth=xgb_cfg.get("max_depth", 4),
        n_estimators=xgb_cfg.get("n_estimators", 120),
        learning_rate=xgb_cfg.get("learning_rate", 0.08),
        subsample=xgb_cfg.get("subsample", 0.9),
        colsample_bytree=xgb_cfg.get("colsample_bytree", 0.8),
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=4,
    )


def train_models(df: pd.DataFrame, cfg: Dict) -> TrainedModels:
    df_feat = engineer_features(df)
    preprocessor, feature_cols = build_preprocessor(df_feat)

    X = df_feat[feature_cols]
    y = df_feat[cfg["data"]["target"]]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg["model"].get("test_size", 0.2), random_state=cfg["model"].get("random_state", 42)
    )

    log_reg = build_logistic(cfg["model"].get("logistic", {}))
    log_pipeline = Pipeline(steps=[("prep", preprocessor), ("model", log_reg)])
    log_pipeline.fit(X_train, y_train)
    y_pred_log = log_pipeline.predict(X_test)
    reports = {"logistic": classification_report(y_test, y_pred_log, zero_division=0)}

    xgb_pipeline = None
    if cfg["model"].get("train_xgboost", True) and XGBClassifier is not None:
        xgb = build_xgb(cfg["model"].get("xgboost", {}))
        xgb_pipeline = Pipeline(steps=[("prep", preprocessor), ("model", xgb)])
        xgb_pipeline.fit(X_train, y_train)
        y_pred_xgb = xgb_pipeline.predict(X_test)
        reports["xgboost"] = classification_report(y_test, y_pred_xgb, zero_division=0)
    elif cfg["model"].get("train_xgboost", True) and XGBClassifier is None:
        logger.warning("XGBoost requested but not installed; skipping.")

    logger.info("Training complete.")
    return TrainedModels(log_pipeline, xgb_pipeline, reports, feature_cols)


def get_inference_pipeline(trained: TrainedModels, prefer_xgb: bool = True) -> Pipeline:
    if prefer_xgb and trained.xgb_pipeline is not None:
        return trained.xgb_pipeline
    return trained.logistic_pipeline

