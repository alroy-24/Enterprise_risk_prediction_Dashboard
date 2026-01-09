import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
from sqlalchemy import create_engine

logger = logging.getLogger(__name__)


@dataclass
class DataSource:
    path: Optional[Path] = None
    postgres_uri: Optional[str] = None
    table: Optional[str] = None


EXPECTED_COLUMNS = {
    "company_id",
    "as_of_date",
    "industry",
    "region",
    "revenue",
    "ebitda",
    "debt",
    "cash",
    "working_capital",
    "compliance_incidents",
    "default_probability",
    "operational_incidents",
    "risk_flag",
}


def load_from_csv(path: Path) -> pd.DataFrame:
    logger.info("Loading CSV from %s", path)
    df = pd.read_csv(path)
    return df


def load_from_excel(path: Path) -> pd.DataFrame:
    logger.info("Loading Excel from %s", path)
    df = pd.read_excel(path)
    return df


def load_from_postgres(uri: str, table: str) -> pd.DataFrame:
    logger.info("Loading Postgres table %s", table)
    engine = create_engine(uri)
    with engine.connect() as conn:
        return pd.read_sql_table(table, con=conn)


def validate_columns(df: pd.DataFrame) -> pd.DataFrame:
    missing = EXPECTED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df


def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["as_of_date"] = pd.to_datetime(df["as_of_date"])
    numeric_cols = [
        "revenue",
        "ebitda",
        "debt",
        "cash",
        "working_capital",
        "compliance_incidents",
        "default_probability",
        "operational_incidents",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_data(source: DataSource) -> pd.DataFrame:
    if source.path and source.path.suffix.lower() == ".csv":
        df = load_from_csv(source.path)
    elif source.path and source.path.suffix.lower() in {".xlsx", ".xls"}:
        df = load_from_excel(source.path)
    elif source.postgres_uri and source.table:
        df = load_from_postgres(source.postgres_uri, source.table)
    else:
        raise ValueError("Provide a CSV/Excel path or Postgres connection.")

    df = validate_columns(df)
    df = coerce_types(df)
    df = df.dropna(subset=["risk_flag"])
    return df


