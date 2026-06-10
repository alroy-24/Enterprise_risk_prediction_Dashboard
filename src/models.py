"""Machine learning models for risk prediction.

This module provides training and inference pipelines for enterprise risk scoring
using Logistic Regression and XGBoost classifiers.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - optional
    XGBClassifier = None  # type: ignore

from features import build_preprocessor, engineer_features
from logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class TrainedModels:
    """Container for trained model pipelines and metadata.
    
    Attributes:
        logistic_pipeline: Trained Logistic Regression pipeline with preprocessor
        xgb_pipeline: Trained XGBoost pipeline (optional, None if not trained)
        reports: Classification reports for each model
        feature_cols: List of feature column names used in training
        metrics: Structured evaluation metrics per model (ROC-AUC, PR-AUC, curves, ...)
        feature_importance: Per-model feature importances aggregated to input columns
        data_summary: Train/test sizes and class balance
    """
    logistic_pipeline: Pipeline
    xgb_pipeline: Optional[Pipeline]
    reports: Dict[str, str]
    feature_cols: List[str]
    metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    feature_importance: Dict[str, List[Dict[str, float]]] = field(default_factory=dict)
    data_summary: Dict[str, Any] = field(default_factory=dict)


def build_logistic(logistic_cfg: Dict) -> LogisticRegression:
    """Build Logistic Regression model from configuration.
    
    Args:
        logistic_cfg: Configuration dictionary with hyperparameters
            - C: Inverse regularization strength (default: 1.0)
            - penalty: Regularization type (default: 'l2')
            - max_iter: Maximum iterations (default: 500)
    
    Returns:
        Configured LogisticRegression instance
    """
    return LogisticRegression(
        C=logistic_cfg.get("C", 1.0),
        penalty=logistic_cfg.get("penalty", "l2"),
        solver="lbfgs",
        max_iter=logistic_cfg.get("max_iter", 500),
        class_weight=logistic_cfg.get("class_weight", "balanced"),
    )


def build_xgb(xgb_cfg: Dict, scale_pos_weight: float = 1.0) -> XGBClassifier:
    """Build XGBoost classifier from configuration.
    
    Args:
        xgb_cfg: Configuration dictionary with hyperparameters
            - max_depth: Maximum tree depth (default: 4)
            - n_estimators: Number of boosting rounds (default: 120)
            - learning_rate: Step size shrinkage (default: 0.08)
            - subsample: Fraction of samples for training (default: 0.9)
            - colsample_bytree: Fraction of features per tree (default: 0.8)
    
    Returns:
        Configured XGBClassifier instance
        
    Raises:
        ImportError: If xgboost package is not installed
    """
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
        scale_pos_weight=xgb_cfg.get("scale_pos_weight", scale_pos_weight),
        random_state=42,
        n_jobs=4,
    )


def _downsample_curve(x: np.ndarray, y: np.ndarray, max_points: int = 200) -> Tuple[List[float], List[float]]:
    """Thin a curve to at most max_points so API payloads stay small."""
    if len(x) <= max_points:
        return [float(v) for v in x], [float(v) for v in y]
    idx = np.linspace(0, len(x) - 1, max_points).astype(int)
    return [float(v) for v in x[idx]], [float(v) for v in y[idx]]


def evaluate_classifier(name: str, pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    """Compute a structured evaluation payload for a fitted pipeline.

    Returns threshold metrics (at 0.5), ranking metrics (ROC-AUC, PR-AUC),
    calibration quality (Brier score + reliability curve) and the raw
    ROC / PR curves, downsampled for transport.
    """
    proba = pipeline.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    fpr, tpr, _ = roc_curve(y_test, proba)
    precision_pts, recall_pts, _ = precision_recall_curve(y_test, proba)
    prob_true, prob_pred = calibration_curve(y_test, proba, n_bins=10, strategy="quantile")
    cm = confusion_matrix(y_test, pred)

    roc_x, roc_y = _downsample_curve(fpr, tpr)
    pr_x, pr_y = _downsample_curve(recall_pts[::-1], precision_pts[::-1])

    metrics = {
        "model": name,
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "pr_auc": float(average_precision_score(y_test, proba)),
        "brier_score": float(brier_score_loss(y_test, proba)),
        "accuracy": float((pred == y_test).mean()),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "confusion_matrix": {
            "tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]), "tp": int(cm[1, 1]),
        },
        "roc_curve": {"fpr": roc_x, "tpr": roc_y},
        "pr_curve": {"recall": pr_x, "precision": pr_y},
        "calibration": {
            "mean_predicted": [float(v) for v in prob_pred],
            "fraction_positive": [float(v) for v in prob_true],
        },
    }
    return metrics


def _group_feature_name(raw_name: str) -> str:
    """Map a transformed feature name back to its input column.

    One-hot columns come out as 'cat__industry_Banks'; numeric passthrough as
    'num__leverage_ratio'. Grouping one-hot levels under their parent column
    keeps the importance chart readable.
    """
    if "__" in raw_name:
        prefix, rest = raw_name.split("__", 1)
        if prefix == "cat":
            return rest.split("_", 1)[0]
        return rest
    return raw_name


def extract_feature_importance(pipeline: Pipeline, top_n: int = 15) -> List[Dict[str, float]]:
    """Pull feature importances (tree models) or |coefficients| (linear models)
    out of a fitted pipeline, aggregated back to input columns."""
    prep = pipeline.named_steps["prep"]
    model = pipeline.named_steps["model"]
    try:
        raw_names = prep.get_feature_names_out()
    except Exception:
        return []

    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        values = np.abs(np.asarray(model.coef_, dtype=float).ravel())
    else:
        return []

    grouped: Dict[str, float] = {}
    for raw, val in zip(raw_names, values):
        key = _group_feature_name(str(raw))
        grouped[key] = grouped.get(key, 0.0) + float(val)

    total = sum(grouped.values()) or 1.0
    ranked = sorted(grouped.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    return [{"feature": k, "importance": v / total} for k, v in ranked]


def train_models(df: pd.DataFrame, cfg: Dict) -> TrainedModels:
    """Train risk prediction models on financial data.
    
    Trains both Logistic Regression (always) and XGBoost (if available and enabled)
    classifiers with proper train/test split and evaluation.
    
    Args:
        df: Input DataFrame with financial features and risk labels
        cfg: Configuration dictionary containing:
            - data.target: Name of target column
            - model.test_size: Test set proportion (default: 0.2)
            - model.random_state: Random seed (default: 42)
            - model.train_xgboost: Whether to train XGBoost (default: True)
            - model.logistic: Logistic Regression hyperparameters
            - model.xgboost: XGBoost hyperparameters
    
    Returns:
        TrainedModels object containing trained pipelines and reports
        
    Example:
        >>> cfg = load_model_config()
        >>> df = pd.read_csv('data/financials.csv')
        >>> trained = train_models(df, cfg)
        >>> print(trained.reports['logistic'])
    """
    logger.info("Starting model training pipeline")
    df_feat = engineer_features(df)
    preprocessor, feature_cols = build_preprocessor(df_feat)
    logger.info(f"Feature engineering complete. Features: {len(feature_cols)}")

    X = df_feat[feature_cols]
    y = df_feat[cfg["data"]["target"]]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg["model"].get("test_size", 0.2),
        random_state=cfg["model"].get("random_state", 42),
        stratify=y,
    )

    pos = int(y_train.sum())
    neg = int(len(y_train) - pos)
    scale_pos_weight = neg / pos if pos else 1.0
    logger.info(f"Class balance — positives: {pos}, negatives: {neg} (scale_pos_weight={scale_pos_weight:.2f})")

    logger.info("Training Logistic Regression model")
    log_reg = build_logistic(cfg["model"].get("logistic", {}))
    log_pipeline = Pipeline(steps=[("prep", preprocessor), ("model", log_reg)])
    log_pipeline.fit(X_train, y_train)
    y_pred_log = log_pipeline.predict(X_test)
    reports = {"logistic": classification_report(y_test, y_pred_log, zero_division=0)}
    metrics = {"logistic": evaluate_classifier("logistic", log_pipeline, X_test, y_test)}
    importance = {"logistic": extract_feature_importance(log_pipeline)}
    logger.info(f"Logistic Regression — ROC-AUC: {metrics['logistic']['roc_auc']:.3f}")

    xgb_pipeline = None
    if cfg["model"].get("train_xgboost", True) and XGBClassifier is not None:
        logger.info("Training XGBoost model")
        xgb = build_xgb(cfg["model"].get("xgboost", {}), scale_pos_weight=scale_pos_weight)
        xgb_pipeline = Pipeline(steps=[("prep", preprocessor), ("model", xgb)])
        xgb_pipeline.fit(X_train, y_train)
        y_pred_xgb = xgb_pipeline.predict(X_test)
        reports["xgboost"] = classification_report(y_test, y_pred_xgb, zero_division=0)
        metrics["xgboost"] = evaluate_classifier("xgboost", xgb_pipeline, X_test, y_test)
        importance["xgboost"] = extract_feature_importance(xgb_pipeline)
        logger.info(f"XGBoost — ROC-AUC: {metrics['xgboost']['roc_auc']:.3f}")
    elif cfg["model"].get("train_xgboost", True) and XGBClassifier is None:
        logger.warning("XGBoost requested but not installed; skipping.")

    data_summary = {
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_features": int(len(feature_cols)),
        "positive_rate": float(y.mean()),
        "target": cfg["data"]["target"],
    }

    logger.info(f"Training complete. Models trained: {list(reports.keys())}")
    return TrainedModels(
        log_pipeline,
        xgb_pipeline,
        reports,
        feature_cols,
        metrics=metrics,
        feature_importance=importance,
        data_summary=data_summary,
    )


def get_inference_pipeline(trained: TrainedModels, prefer_xgb: bool = True) -> Pipeline:
    """Get inference pipeline for predictions.
    
    Args:
        trained: TrainedModels object containing trained pipelines
        prefer_xgb: If True and XGBoost is available, use XGBoost; otherwise use Logistic Regression
    
    Returns:
        Selected sklearn Pipeline for making predictions
    """
    if prefer_xgb and trained.xgb_pipeline is not None:
        logger.debug("Using XGBoost pipeline for inference")
        return trained.xgb_pipeline
    logger.debug("Using Logistic Regression pipeline for inference")
    return trained.logistic_pipeline

