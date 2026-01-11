"""
Market Data & Time-Series Module
Simulates equity prices, fixed income yields, spreads, and risk factors (VIX-style).
For industry-standard market risk analysis.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional


def generate_market_data(n_companies: int = 20, start_date: str = "2024-01-01", end_date: str = "2024-12-31") -> pd.DataFrame:
    """
    Generate synthetic market data (equity prices, yields, spreads) for time-series analysis.
    Industry-standard market data structure.
    """
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    companies = [f"C{i:03d}" for i in range(1, n_companies + 1)]
    
    data = []
    np.random.seed(42)
    
    for company in companies:
        # Base equity price (random walk)
        base_price = np.random.uniform(50, 200)
        prices = [base_price]
        returns = np.random.normal(0.0005, 0.02, len(dates) - 1)  # Daily returns
        
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))
        
        # Bond yields (inverted to prices)
        base_yield = np.random.uniform(0.02, 0.06)  # 2-6% yield
        yields = base_yield + np.random.normal(0, 0.001, len(dates))
        yields = np.clip(yields, 0.01, 0.10)
        
        # Credit spreads (over risk-free rate)
        base_spread = np.random.uniform(0.005, 0.05)  # 50-500 bps
        spreads = base_spread + np.random.normal(0, 0.002, len(dates))
        spreads = np.clip(spreads, 0, 0.10)
        
        # VIX-style volatility (company-specific)
        base_vol = np.random.uniform(0.15, 0.35)  # 15-35% volatility
        vols = base_vol + np.random.normal(0, 0.02, len(dates))
        vols = np.clip(vols, 0.10, 0.50)
        
        for i, date in enumerate(dates):
            data.append({
                "company_id": company,
                "date": date,
                "equity_price": prices[i],
                "yield": yields[i],
                "credit_spread": spreads[i],
                "volatility": vols[i],
                "volume": np.random.uniform(100000, 10000000),
            })
    
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    return df


def calculate_returns(df: pd.DataFrame, price_col: str = "equity_price") -> pd.DataFrame:
    """Calculate daily returns, log returns, and rolling metrics."""
    df = df.sort_values(["company_id", "date"]).copy()
    df = df.reset_index(drop=True)  # Reset index to avoid alignment issues
    
    # Daily returns
    df["daily_return"] = df.groupby("company_id")[price_col].pct_change()
    
    # Log returns
    shifted_price = df.groupby("company_id")[price_col].shift(1)
    df["log_return"] = np.log(df[price_col] / shifted_price)
    
    # Cumulative returns (using transform to maintain index alignment)
    df["cumulative_return"] = df.groupby("company_id")["daily_return"].transform(
        lambda x: (1 + x.fillna(0)).cumprod() - 1
    )
    
    # Rolling metrics with proper error handling
    def safe_rolling_vol(x):
        result = x.rolling(30, min_periods=5).std() * np.sqrt(252)
        return result
    
    def safe_rolling_sharpe(x):
        mean_val = x.rolling(30, min_periods=5).mean() * 252
        std_val = x.rolling(30, min_periods=5).std() * np.sqrt(252)
        result = mean_val / (std_val + 1e-6)  # Avoid division by zero
        return result
    
    df["rolling_vol_30d"] = df.groupby("company_id")["daily_return"].transform(safe_rolling_vol)
    df["rolling_sharpe_30d"] = df.groupby("company_id")["daily_return"].transform(safe_rolling_sharpe)
    
    return df


def merge_market_with_financials(market_df: pd.DataFrame, financial_df: pd.DataFrame) -> pd.DataFrame:
    """Merge market data with financial statements by company_id and date."""
    # Get latest market data for each company
    latest_market = market_df.groupby("company_id").last().reset_index()
    
    # Merge on company_id
    merged = financial_df.merge(latest_market, on="company_id", how="left", suffixes=("_fin", "_mkt"))
    
    return merged

