"""
Value at Risk (VaR) Calculations
Implements parametric, historical, and Monte Carlo VaR methods.
Industry-standard risk metrics for professional risk analysis.
"""
import pandas as pd
import numpy as np
from typing import Literal, Optional
from scipy import stats


def parametric_var(returns: pd.Series, confidence_level: float = 0.95, holding_period: int = 1) -> float:
    """
    Parametric VaR (Variance-Covariance method).
    
    Assumes returns are normally distributed.
    VaR = -mean(returns) * holding_period - z_score * std(returns) * sqrt(holding_period)
    """
    mean_return = returns.mean()
    std_return = returns.std()
    
    # Z-score for confidence level (e.g., 1.65 for 95%, 2.33 for 99%)
    z_score = stats.norm.ppf(confidence_level)
    
    var = -(mean_return * holding_period - z_score * std_return * np.sqrt(holding_period))
    return var


def historical_var(returns: pd.Series, confidence_level: float = 0.95) -> float:
    """
    Historical VaR (Historical Simulation).
    
    Uses empirical distribution of historical returns.
    """
    var_percentile = (1 - confidence_level) * 100
    var = -np.percentile(returns.dropna(), var_percentile)
    return var


def monte_carlo_var(
    returns: pd.Series,
    confidence_level: float = 0.95,
    n_simulations: int = 10000,
    holding_period: int = 1
) -> float:
    """
    Monte Carlo VaR.
    
    Simulates future returns using historical distribution.
    """
    mean_return = returns.mean()
    std_return = returns.std()
    
    # Generate random samples
    simulated_returns = np.random.normal(mean_return, std_return, (n_simulations, holding_period))
    cumulative_returns = np.sum(simulated_returns, axis=1)
    
    var_percentile = (1 - confidence_level) * 100
    var = -np.percentile(cumulative_returns, var_percentile)
    return var


def conditional_var(returns: pd.Series, confidence_level: float = 0.95) -> float:
    """
    Conditional VaR (CVaR / Expected Shortfall).
    
    Expected loss given that loss exceeds VaR threshold.
    """
    var = historical_var(returns, confidence_level)
    tail_losses = returns[returns <= -var]
    
    if len(tail_losses) > 0:
        cvar = -tail_losses.mean()
    else:
        cvar = var  # Fallback
    
    return cvar


def calculate_portfolio_var(
    returns_df: pd.DataFrame,
    weights: Optional[pd.Series] = None,
    method: Literal["parametric", "historical", "monte_carlo"] = "parametric",
    confidence_level: float = 0.95,
    holding_period: int = 1
) -> dict:
    """
    Calculate portfolio-level VaR.
    
    If weights provided, calculates weighted portfolio returns first.
    """
    if weights is None:
        # Equal weights
        weights = pd.Series(1.0 / len(returns_df.columns), index=returns_df.columns)
    
    # Portfolio returns
    portfolio_returns = (returns_df * weights).sum(axis=1)
    
    # Calculate VaR using selected method
    if method == "parametric":
        var = parametric_var(portfolio_returns, confidence_level, holding_period)
    elif method == "historical":
        var = historical_var(portfolio_returns, confidence_level)
    elif method == "monte_carlo":
        var = monte_carlo_var(portfolio_returns, confidence_level, holding_period=holding_period)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    cvar = conditional_var(portfolio_returns, confidence_level)
    
    return {
        "var": var,
        "cvar": cvar,
        "confidence_level": confidence_level,
        "method": method,
        "portfolio_return_mean": portfolio_returns.mean(),
        "portfolio_return_std": portfolio_returns.std(),
    }


def calculate_individual_var(
    returns_df: pd.DataFrame,
    method: Literal["parametric", "historical"] = "parametric",
    confidence_level: float = 0.95
) -> pd.DataFrame:
    """
    Calculate VaR for each asset/company individually.
    """
    results = []
    
    for col in returns_df.columns:
        returns = returns_df[col].dropna()
        if len(returns) < 10:  # Need sufficient data
            continue
        
        if method == "parametric":
            var = parametric_var(returns, confidence_level)
        else:
            var = historical_var(returns, confidence_level)
        
        cvar = conditional_var(returns, confidence_level)
        
        results.append({
            "company_id": col,
            "var": var,
            "cvar": cvar,
            "mean_return": returns.mean(),
            "std_return": returns.std(),
            "confidence_level": confidence_level,
        })
    
    return pd.DataFrame(results)

