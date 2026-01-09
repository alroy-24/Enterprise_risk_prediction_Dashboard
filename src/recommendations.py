import pandas as pd


def generate_recommendations(row: pd.Series) -> list[str]:
    recs = []
    if row.get("leverage_ratio", 0) > 3:
        recs.append("Consider deleveraging or refinancing to reduce leverage ratio.")
    if row.get("liquidity_ratio", 0) < 0.3:
        recs.append("Increase liquidity buffer via cash conservation or credit lines.")
    if row.get("compliance_intensity", 0) > 0.8:
        recs.append("Prioritize compliance remediation and board-level reporting.")
    if row.get("predicted_proba", 0) > 0.5:
        recs.append("Implement heightened monitoring; run monthly scenario reviews.")
    if not recs:
        recs.append("Risk posture stable; continue monitoring key indicators.")
    return recs


