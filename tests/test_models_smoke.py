import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_ingest import DataSource, load_data  # noqa: E402
from models import train_models, get_inference_pipeline  # noqa: E402
import config as cfg_mod  # noqa: E402


def test_training_and_inference_smoke():
    cfg = cfg_mod.load_model_config()
    source = cfg["data"]["source"]
    ds = DataSource(path=ROOT / source)
    df = load_data(ds)

    trained = train_models(df, cfg)
    pipeline = get_inference_pipeline(trained)

    assert trained.logistic_pipeline is not None
    proba = pipeline.predict_proba(
        df[trained.feature_cols]
    )  # uses same feature cols as training
    assert proba.shape[0] == len(df)



