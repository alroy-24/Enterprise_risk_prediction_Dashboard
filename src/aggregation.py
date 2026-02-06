import pandas as pd
from typing import Dict


def aggregate_scores(df: pd.DataFrame, weights: Dict) -> pd.DataFrame:
    df = df.copy()
    df["financial_score"] = df["predicted_proba"] * 0.6 + df["leverage_ratio"].rank(pct=True) * 0.4
    df["operational_score"] = df["incident_rate"].rank(pct=True) * 0.7 + df["default_probability"] * 0.3
    df["compliance_score"] = df["compliance_intensity"].rank(pct=True)

    df["aggregated_risk"] = (
        df["financial_score"] * weights["financial"]
        + df["operational_score"] * weights["operational"]
        + df["compliance_score"] * weights["compliance"]
    )
    return df


