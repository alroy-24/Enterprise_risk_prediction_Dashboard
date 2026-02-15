"""
MLflow integration for experiment tracking and model registry.
Provides utilities for logging metrics, parameters, and artifacts.
"""
from typing import Any, Dict, Optional
import mlflow
from mlflow.models import infer_signature
import pandas as pd
from sklearn.pipeline import Pipeline

from logging_config import get_logger

logger = get_logger(__name__)


class MLflowTracker:
    """MLflow experiment tracking wrapper."""
    
    def __init__(self, tracking_uri: str, experiment_name: str):
        """
        Initialize MLflow tracker.
        
        Args:
            tracking_uri: MLflow tracking server URI or local path
            experiment_name: Name of the experiment
        """
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
            logger.info(f"MLflow tracking URI set to: {tracking_uri}")
        
        mlflow.set_experiment(experiment_name)
        logger.info(f"MLflow experiment set to: {experiment_name}")
    
    def start_run(self, run_name: Optional[str] = None) -> mlflow.ActiveRun:
        """
        Start a new MLflow run.
        
        Args:
            run_name: Optional name for the run
            
        Returns:
            Active MLflow run context
        """
        return mlflow.start_run(run_name=run_name)
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """
        Log parameters to current run.
        
        Args:
            params: Dictionary of parameters to log
        """
        mlflow.log_params(params)
        logger.debug(f"Logged {len(params)} parameters to MLflow")
    
    def log_metrics(self, metrics: Dict[str, float]) -> None:
        """
        Log metrics to current run.
        
        Args:
            metrics: Dictionary of metrics to log
        """
        mlflow.log_metrics(metrics)
        logger.debug(f"Logged {len(metrics)} metrics to MLflow")
    
    def log_model(
        self,
        model: Pipeline,
        artifact_path: str,
        X_sample: Optional[pd.DataFrame] = None,
        y_sample: Optional[pd.Series] = None
    ) -> None:
        """
        Log sklearn model to MLflow.
        
        Args:
            model: Trained sklearn pipeline
            artifact_path: Path within run artifacts to save model
            X_sample: Sample input for signature inference
            y_sample: Sample output for signature inference
        """
        signature = None
        if X_sample is not None and y_sample is not None:
            signature = infer_signature(X_sample, y_sample)
        
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=artifact_path,
            signature=signature
        )
        logger.info(f"Model logged to MLflow at: {artifact_path}")
    
    def log_artifact(self, local_path: str) -> None:
        """
        Log file artifact to current run.
        
        Args:
            local_path: Path to local file to log
        """
        mlflow.log_artifact(local_path)
        logger.debug(f"Logged artifact: {local_path}")
    
    def log_dict(self, dictionary: Dict, filename: str) -> None:
        """
        Log dictionary as JSON artifact.
        
        Args:
            dictionary: Dictionary to log
            filename: Name for the JSON file
        """
        mlflow.log_dict(dictionary, filename)
        logger.debug(f"Logged dictionary as: {filename}")
    
    def set_tags(self, tags: Dict[str, str]) -> None:
        """
        Set tags on current run.
        
        Args:
            tags: Dictionary of tags to set
        """
        mlflow.set_tags(tags)
        logger.debug(f"Set {len(tags)} tags on run")


def get_mlflow_tracker(config: Dict) -> Optional[MLflowTracker]:
    """
    Create MLflow tracker from configuration.
    
    Args:
        config: Configuration dictionary with governance settings
        
    Returns:
        MLflowTracker if enabled, None otherwise
    """
    if not config.get("model", {}).get("use_mlflow", False):
        logger.info("MLflow tracking disabled in config")
        return None
    
    governance = config.get("governance", {})
    tracking_uri = governance.get("mlflow_tracking_uri", "mlruns")
    experiment_name = governance.get("mlflow_experiment_name", "risk_prediction")
    
    try:
        tracker = MLflowTracker(tracking_uri, experiment_name)
        logger.info("MLflow tracker initialized successfully")
        return tracker
    except Exception as e:
        logger.error(f"Failed to initialize MLflow tracker: {e}")
        return None
