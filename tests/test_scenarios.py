import sys
from pathlib import Path

import pandas as pd


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from scenarios import apply_shock  # noqa: E402


def test_apply_shock_multiplicative_and_additive():
    df = pd.DataFrame(
        {
            "revenue": [100.0],
            "debt": [50.0],
            "compliance_incidents": [1],
        }
    )

    shocked = apply_shock(
        df,
        {
            "revenue": -0.1,  # -10% multiplicative
            "debt": 0.2,  # +20% multiplicative
            "compliance_incidents": 1,  # +1 absolute
        },
    )

    assert shocked.loc[0, "revenue"] == 90.0
    assert shocked.loc[0, "debt"] == 60.0
    assert shocked.loc[0, "compliance_incidents"] == 2



