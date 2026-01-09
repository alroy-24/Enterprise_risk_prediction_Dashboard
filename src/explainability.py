import logging
from typing import Optional

import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


def compute_shap_values(pipeline: Pipeline, X: pd.DataFrame) -> tuple[np.ndarray, shap.Explanation]:
    model = pipeline.named_steps["model"]
    prep = pipeline.named_steps["prep"]

    # Transform features for the model
    X_transformed = prep.transform(X)
    explainer = shap.Explainer(model, X_transformed)
    shap_values = explainer(X_transformed)
    return shap_values.values, shap_values


def summarize_top_features(shap_values: shap.Explanation, feature_names: list[str], top_n: int = 5) -> pd.DataFrame:
    mean_abs = np.abs(shap_values.values).mean(axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:top_n]
    return pd.DataFrame(
        {
            "feature": np.array(feature_names)[top_idx],
            "mean_abs_shap": mean_abs[top_idx],
        }
    )


def explain_single(pipeline: Pipeline, X_row: pd.DataFrame) -> Optional[shap.Explanation]:
    try:
        _, shap_values = compute_shap_values(pipeline, X_row)
        return shap_values
    except Exception as exc:  # pragma: no cover - SHAP can be brittle on tiny sets
        logger.warning("SHAP explanation failed: %s", exc)
        return None


