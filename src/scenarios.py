import pandas as pd
from typing import Dict


def apply_shock(df: pd.DataFrame, shock_cfg: Dict[str, float]) -> pd.DataFrame:
    df_s = df.copy()
    for feature, delta in shock_cfg.items():
        if feature in df_s.columns:
            if abs(delta) < 1 and feature not in {"compliance_incidents", "operational_incidents"}:
                df_s[feature] = df_s[feature] * (1 + delta)
            else:
                df_s[feature] = df_s[feature] + delta
        elif feature == "ebitda_margin":
            df_s["ebitda"] = df_s["ebitda"] * (1 + delta)
        else:
            # unknown feature; ignore silently for PoC
            continue
    return df_s


def run_scenarios(df: pd.DataFrame, scenarios: Dict, infer_fn):
    results = {}
    for name, cfg in scenarios["scenarios"].items():
        shocked = apply_shock(df, cfg.get("shocks", {}))
        proba = infer_fn(shocked)
        results[name] = {"description": cfg.get("description", ""), "probability": proba}
    return results


