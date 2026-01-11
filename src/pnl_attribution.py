"""
PnL Attribution Analysis
Break down portfolio performance by factors (equity, fixed income, credit, FX, etc.).
Industry-standard performance analysis.
"""
import pandas as pd
import numpy as np
from typing import Optional


def calculate_pnl(
    prices: pd.DataFrame,
    positions: Optional[pd.Series] = None,
    initial_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Calculate PnL (Profit and Loss) for each asset.
    
    PnL = (End Price - Start Price) * Position
    """
    if "company_id" not in prices.columns or "equity_price" not in prices.columns:
        return pd.DataFrame(columns=["company_id", "start_price", "end_price", "position", "pnl", "pnl_pct"])
    
    # Ensure date column exists and is datetime
    if "date" not in prices.columns:
        return pd.DataFrame(columns=["company_id", "start_price", "end_price", "position", "pnl", "pnl_pct"])
    
    prices = prices.copy()
    if not pd.api.types.is_datetime64_any_dtype(prices["date"]):
        prices["date"] = pd.to_datetime(prices["date"])
    
    # Get start prices
    if initial_date:
        start_df = prices[prices["date"] == pd.to_datetime(initial_date)]
    else:
        start_df = prices.sort_values("date").groupby("company_id").first().reset_index()
    
    start_prices = start_df.set_index("company_id")["equity_price"]
    
    # Get end prices
    if end_date:
        end_df = prices[prices["date"] == pd.to_datetime(end_date)]
    else:
        end_df = prices.sort_values("date").groupby("company_id").last().reset_index()
    
    end_prices = end_df.set_index("company_id")["equity_price"]
    
    # Default positions
    if positions is None:
        positions = pd.Series(1.0, index=start_prices.index)
    
    # Align indices
    common_idx = start_prices.index.intersection(end_prices.index)
    if len(common_idx) == 0:
        return pd.DataFrame(columns=["company_id", "start_price", "end_price", "position", "pnl", "pnl_pct"])
    
    start_prices = start_prices[common_idx]
    end_prices = end_prices[common_idx]
    positions = positions[common_idx] if common_idx.isin(positions.index).any() else pd.Series(1.0, index=common_idx)
    
    # Calculate PnL
    pnl = (end_prices - start_prices) * positions
    pnl_pct = (end_prices / (start_prices + 1e-6) - 1) * 100  # Avoid division by zero
    
    results = pd.DataFrame({
        "company_id": common_idx.values,
        "start_price": start_prices.values,
        "end_price": end_prices.values,
        "position": positions.values if isinstance(positions, pd.Series) else positions,
        "pnl": pnl.values,
        "pnl_pct": pnl_pct.values,
    })
    
    return results


def attribute_pnl_by_factor(
    pnl_df: pd.DataFrame,
    market_df: pd.DataFrame,
    risk_factors: list[str] = ["equity_price", "yield", "credit_spread", "volatility"]
) -> pd.DataFrame:
    """
    Attribute PnL to different risk factors.
    
    Uses regression to decompose PnL into factor contributions.
    """
    results = []
    
    for company_id in pnl_df["company_id"]:
        company_market = market_df[market_df["company_id"] == company_id].sort_values("date")
        company_pnl = pnl_df[pnl_df["company_id"] == company_id]["pnl"].values[0]
        
        if len(company_market) < 10:
            continue
        
        # Calculate factor changes
        factor_changes = {}
        for factor in risk_factors:
            if factor in company_market.columns:
                factor_changes[factor] = company_market[factor].iloc[-1] - company_market[factor].iloc[0]
        
        # Simple attribution (can be enhanced with regression)
        # For now, use correlation-based attribution
        pnl_attribution = {}
        total_change = sum(abs(v) for v in factor_changes.values())
        
        if total_change > 0:
            for factor, change in factor_changes.items():
                # Weight by magnitude of change (simplified)
                pnl_attribution[f"{factor}_attribution"] = company_pnl * (abs(change) / total_change) * np.sign(change)
        else:
            for factor in risk_factors:
                pnl_attribution[f"{factor}_attribution"] = 0.0
        
        results.append({
            "company_id": company_id,
            "total_pnl": company_pnl,
            **pnl_attribution
        })
    
    return pd.DataFrame(results)


def calculate_portfolio_pnl(
    pnl_df: pd.DataFrame,
    weights: Optional[pd.Series] = None
) -> dict:
    """
    Calculate aggregate portfolio PnL.
    """
    if weights is None:
        weights = pd.Series(1.0 / len(pnl_df), index=pnl_df["company_id"])
    
    weighted_pnl = (pnl_df.set_index("company_id")["pnl"] * weights).sum()
    total_pnl_pct = (pnl_df.set_index("company_id")["pnl_pct"] * weights).sum()
    
    return {
        "total_pnl": weighted_pnl,
        "total_pnl_pct": total_pnl_pct,
        "num_positions": len(pnl_df),
        "best_performer": pnl_df.loc[pnl_df["pnl"].idxmax(), "company_id"],
        "worst_performer": pnl_df.loc[pnl_df["pnl"].idxmin(), "company_id"],
    }

