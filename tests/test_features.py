import sys
from pathlib import Path

import pandas as pd


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from features import engineer_features  # noqa: E402


def test_engineer_features_creates_expected_columns():
    df = pd.DataFrame(
        {
            "company_id": ["C1"],
            "as_of_date": ["2024-12-31"],
            "industry": ["Software"],
            "region": ["NA"],
            "revenue": [1_000_000],
            "ebitda": [200_000],
            "debt": [300_000],
            "cash": [100_000],
            "working_capital": [150_000],
            "compliance_incidents": [0],
            "default_probability": [0.02],
            "operational_incidents": [1],
            "risk_flag": [0],
        }
    )

    df_feat = engineer_features(df)

    for col in [
        "leverage_ratio",
        "liquidity_ratio",
        "ebitda_margin",
        "default_buffer",
        "incident_rate",
        "compliance_intensity",
    ]:
        assert col in df_feat.columns
        assert df_feat[col].notna().all()



