from contextlib import contextmanager
from typing import Dict, Optional


@contextmanager
def mlflow_run(enabled: bool, uri: str, run_name: str = "erip-run"):
    if not enabled or not uri:
        yield None
        return
    import mlflow

    mlflow.set_tracking_uri(uri)
    with mlflow.start_run(run_name=run_name) as run:
        yield run


def log_params(enabled: bool, params: Dict, uri: str):
    if not enabled or not uri:
        return
    import mlflow

    mlflow.log_params(params)


def log_metrics(enabled: bool, metrics: Dict, uri: str):
    if not enabled or not uri:
        return
    import mlflow

    mlflow.log_metrics(metrics)


