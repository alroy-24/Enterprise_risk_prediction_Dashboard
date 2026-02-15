"""Machine learning models for risk prediction.

This module provides training and inference pipelines for enterprise risk scoring
using Logistic Regression and XGBoost classifiers.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
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
    """
    logistic_pipeline: Pipeline
    xgb_pipeline: Optional[Pipeline]
    reports: Dict[str, str]
    feature_cols: List[str]


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
    )


def build_xgb(xgb_cfg: Dict) -> XGBClassifier:
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
        random_state=42,
        n_jobs=4,
    )


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
        X, y, test_size=cfg["model"].get("test_size", 0.2), random_state=cfg["model"].get("random_state", 42)
    )

    logger.info("Training Logistic Regression model")
    log_reg = build_logistic(cfg["model"].get("logistic", {}))
    log_pipeline = Pipeline(steps=[("prep", preprocessor), ("model", log_reg)])
    log_pipeline.fit(X_train, y_train)
    y_pred_log = log_pipeline.predict(X_test)
    reports = {"logistic": classification_report(y_test, y_pred_log, zero_division=0)}
    logger.info(f"Logistic Regression training complete. Test accuracy: {(y_pred_log == y_test).mean():.3f}")

    xgb_pipeline = None
    if cfg["model"].get("train_xgboost", True) and XGBClassifier is not None:
        logger.info("Training XGBoost model")
        xgb = build_xgb(cfg["model"].get("xgboost", {}))
        xgb_pipeline = Pipeline(steps=[("prep", preprocessor), ("model", xgb)])
        xgb_pipeline.fit(X_train, y_train)
        y_pred_xgb = xgb_pipeline.predict(X_test)
        reports["xgboost"] = classification_report(y_test, y_pred_xgb, zero_division=0)
        logger.info(f"XGBoost training complete. Test accuracy: {(y_pred_xgb == y_test).mean():.3f}")
    elif cfg["model"].get("train_xgboost", True) and XGBClassifier is None:
        logger.warning("XGBoost requested but not installed; skipping.")

    logger.info(f"Training complete. Models trained: {list(reports.keys())}")
    return TrainedModels(log_pipeline, xgb_pipeline, reports, feature_cols)


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

