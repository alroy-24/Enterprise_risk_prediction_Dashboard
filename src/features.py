import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer enterprise-grade risk features including industry-standard financial ratios.
    """
    df = df.copy()
    
    # Ensure we have required columns with defaults
    if "total_assets" not in df.columns:
        # Estimate total assets from available data
        df["total_assets"] = df.get("total_assets", df.get("revenue", 0) * 1.5)
    if "total_equity" not in df.columns:
        df["total_equity"] = df.get("total_equity", df.get("total_assets", 0) - df.get("debt", 0))
    if "current_assets" not in df.columns:
        df["current_assets"] = df.get("current_assets", df.get("cash", 0) + df.get("working_capital", 0))
    if "current_liabilities" not in df.columns:
        df["current_liabilities"] = df.get("current_liabilities", df.get("current_assets", 0) - df.get("working_capital", 0))
    if "interest_expense" not in df.columns:
        # Estimate interest expense as ~3-5% of debt
        df["interest_expense"] = df.get("interest_expense", df.get("debt", 0) * 0.04)
    if "accounts_receivable" not in df.columns:
        # Estimate AR as ~15% of revenue (typical DSO ~55 days)
        df["accounts_receivable"] = df.get("accounts_receivable", df.get("revenue", 0) * 0.15)
    if "net_income" not in df.columns:
        # Estimate net income from EBITDA (rough approximation)
        df["net_income"] = df.get("net_income", df.get("ebitda", 0) * 0.6)
    
    # === LEVERAGE RATIOS ===
    # Debt-to-EBITDA (standard leverage metric)
    df["leverage_ratio"] = df["debt"] / (df["ebitda"].abs() + 1e-6)
    # Debt-to-Equity (capital structure)
    df["debt_to_equity"] = df["debt"] / (df["total_equity"].abs() + 1e-6)
    # Debt-to-Assets (asset coverage)
    df["debt_to_assets"] = df["debt"] / (df["total_assets"].abs() + 1e-6)
    
    # === LIQUIDITY RATIOS ===
    # Current Ratio (current assets / current liabilities)
    df["current_ratio"] = df["current_assets"] / (df["current_liabilities"].abs() + 1e-6)
    # Quick Ratio (cash + AR) / current liabilities
    df["quick_ratio"] = (df["cash"] + df["accounts_receivable"]) / (df["current_liabilities"].abs() + 1e-6)
    # Cash Ratio (cash / current liabilities)
    df["cash_ratio"] = df["cash"] / (df["current_liabilities"].abs() + 1e-6)
    # Working Capital Ratio
    df["liquidity_ratio"] = df["cash"] / (df["working_capital"].abs() + 1e-6)
    
    # === PROFITABILITY RATIOS ===
    df["ebitda_margin"] = df["ebitda"] / (df["revenue"].abs() + 1e-6)
    df["net_margin"] = df["net_income"] / (df["revenue"].abs() + 1e-6)
    df["roa"] = df["net_income"] / (df["total_assets"].abs() + 1e-6)  # Return on Assets
    df["roe"] = df["net_income"] / (df["total_equity"].abs() + 1e-6)  # Return on Equity
    
    # === COVERAGE RATIOS ===
    # Interest Coverage (EBITDA / interest expense)
    df["interest_coverage"] = df["ebitda"] / (df["interest_expense"].abs() + 1e-6)
    # EBITDA Coverage (EBITDA / (interest + principal))
    df["ebitda_coverage"] = df["ebitda"] / (df["interest_expense"].abs() + df["debt"] * 0.1 + 1e-6)
    
    # === EFFICIENCY RATIOS ===
    # Days Sales Outstanding (DSO) - AR / (Revenue/365)
    df["dso"] = (df["accounts_receivable"] / (df["revenue"].abs() / 365 + 1e-6)).clip(0, 365)
    # Asset Turnover
    df["asset_turnover"] = df["revenue"] / (df["total_assets"].abs() + 1e-6)
    
    # === ALTMAN Z-SCORE (Bankruptcy Prediction) ===
    # Z = 1.2*(Working Capital/Assets) + 1.4*(Retained Earnings/Assets) + 3.3*(EBIT/Assets) + 0.6*(Market Value Equity/Debt) + 1.0*(Sales/Assets)
    # Simplified version using available data
    retained_earnings = df.get("retained_earnings", df.get("total_equity", 0) * 0.5)
    wc_assets = df["working_capital"] / (df["total_assets"].abs() + 1e-6)
    re_assets = retained_earnings / (df["total_assets"].abs() + 1e-6)
    ebit_assets = df["ebitda"] / (df["total_assets"].abs() + 1e-6)  # Approximate EBIT with EBITDA
    equity_debt = df["total_equity"] / (df["debt"].abs() + 1e-6)
    sales_assets = df["revenue"] / (df["total_assets"].abs() + 1e-6)
    
    df["altman_z_score"] = (
        1.2 * wc_assets +
        1.4 * re_assets +
        3.3 * ebit_assets +
        0.6 * equity_debt +
        1.0 * sales_assets
    )
    
    # === OPERATIONAL RISK METRICS ===
    df["default_buffer"] = 1 - df["default_probability"]
    df["incident_rate"] = df["operational_incidents"] / (df["revenue"].abs() + 1e-6)
    # Normalize compliance incidents by revenue scale
    df["compliance_intensity"] = df["compliance_incidents"] / (df["revenue"].abs() / 1e6 + 1e-3)
    
    # === SIZE & SCALE METRICS ===
    df["revenue_log"] = np.log1p(df["revenue"].abs())
    df["asset_log"] = np.log1p(df["total_assets"].abs())
    
    return df


def build_preprocessor(df: pd.DataFrame):
    """
    Build preprocessing pipeline with enterprise risk features.
    """
    # Core financial ratios (most predictive)
    core_ratios = [
        "leverage_ratio",
        "debt_to_equity",
        "current_ratio",
        "quick_ratio",
        "interest_coverage",
        "ebitda_margin",
        "altman_z_score",
        "roa",
    ]
    
    # Additional risk metrics
    risk_metrics = [
        "default_probability",
        "default_buffer",
        "incident_rate",
        "compliance_intensity",
        "dso",
    ]
    
    # Raw financials (scaled)
    raw_financials = [
        "revenue",
        "ebitda",
        "debt",
        "cash",
        "working_capital",
    ]
    
    numeric_cols = core_ratios + risk_metrics + raw_financials
    
    # Filter to only columns that exist in the dataframe
    numeric_cols = [col for col in numeric_cols if col in df.columns]
    
    categorical_cols = ["industry", "region"]
    categorical_cols = [col for col in categorical_cols if col in df.columns]

    # For now we just pass numeric features through; could add scaling here.
    numeric_transformer = "passthrough"
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical_cols),
            ("num", numeric_transformer, numeric_cols),
        ]
    )
    feature_cols = categorical_cols + numeric_cols
    return preprocessor, feature_cols

