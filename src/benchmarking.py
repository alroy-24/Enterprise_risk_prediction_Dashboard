"""
Peer Benchmarking - Compare companies to industry/region averages.
Enterprise risk analytics standard practice.
"""
import pandas as pd
import numpy as np


def compute_peer_benchmarks(df: pd.DataFrame, group_by: str = "industry") -> pd.DataFrame:
    """
    Compute industry/region benchmarks for key risk metrics.
    
    Returns dataframe with benchmark statistics per group.
    """
    metrics = [
        "leverage_ratio", "current_ratio", "interest_coverage",
        "ebitda_margin", "altman_z_score", "predicted_proba",
        "debt_to_equity", "roa"
    ]
    
    # Filter to metrics that exist
    metrics = [m for m in metrics if m in df.columns]
    
    if not metrics:
        return pd.DataFrame()
    
    # Compute aggregations - use list of functions for each metric
    agg_functions = ["mean", "median", "std"]
    
    # Build aggregation dictionary
    agg_dict = {metric: agg_functions for metric in metrics}
    
    benchmarks = df.groupby(group_by)[metrics].agg(agg_dict).round(4)
    
    # Add quantiles separately (pandas doesn't support q25/q75 directly in agg)
    for metric in metrics:
        q25_vals = df.groupby(group_by)[metric].quantile(0.25)
        q75_vals = df.groupby(group_by)[metric].quantile(0.75)
        benchmarks[(metric, "q25")] = q25_vals
        benchmarks[(metric, "q75")] = q75_vals
    
    # Flatten column names
    benchmarks.columns = [f"{col[0]}_{col[1]}" for col in benchmarks.columns]
    
    return benchmarks


def add_peer_comparison(df: pd.DataFrame, group_by: str = "industry") -> pd.DataFrame:
    """
    Add peer comparison flags (above/below industry average).
    """
    df = df.copy()
    
    if group_by not in df.columns:
        return df
    
    metrics_to_compare = [
        "leverage_ratio", "current_ratio", "interest_coverage",
        "ebitda_margin", "altman_z_score", "predicted_proba"
    ]
    metrics_to_compare = [m for m in metrics_to_compare if m in df.columns]
    
    for metric in metrics_to_compare:
        industry_avg = df.groupby(group_by)[metric].transform("mean")
        df[f"{metric}_vs_peer"] = df[metric] - industry_avg
        df[f"{metric}_percentile"] = df.groupby(group_by)[metric].transform(
            lambda x: pd.qcut(x.rank(method="first"), q=4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
        )
    
    return df


def get_peer_summary(df: pd.DataFrame, company_id: str, group_by: str = "industry") -> dict:
    """
    Get peer comparison summary for a specific company.
    """
    if company_id not in df["company_id"].values:
        return {}
    
    company = df[df["company_id"] == company_id].iloc[0]
    industry = company.get(group_by, "Unknown")
    peers = df[df[group_by] == industry]
    
    if len(peers) < 2:
        return {"message": "Insufficient peers for comparison"}
    
    summary = {
        "company_id": company_id,
        "industry": industry,
        "peer_count": len(peers) - 1,
    }
    
    metrics = ["leverage_ratio", "current_ratio", "ebitda_margin", "predicted_proba"]
    metrics = [m for m in metrics if m in df.columns]
    
    for metric in metrics:
        company_val = company[metric]
        peer_avg = peers[peers["company_id"] != company_id][metric].mean()
        peer_median = peers[peers["company_id"] != company_id][metric].median()
        
        summary[f"{metric}_company"] = round(company_val, 4)
        summary[f"{metric}_peer_avg"] = round(peer_avg, 4)
        summary[f"{metric}_peer_median"] = round(peer_median, 4)
        summary[f"{metric}_vs_avg"] = round(company_val - peer_avg, 4)
        summary[f"{metric}_percentile"] = (
            "Top 25%" if company_val <= peers[metric].quantile(0.25) else
            "Top 50%" if company_val <= peers[metric].quantile(0.50) else
            "Top 75%" if company_val <= peers[metric].quantile(0.75) else "Bottom 25%"
        )
    
    return summary

