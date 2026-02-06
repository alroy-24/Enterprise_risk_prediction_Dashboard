"""
Enhanced data ingestion module supporting both sample and real-world data sources.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
import yaml

# Import real-world data loaders
try:
    from real_world_data_loaders import YahooFinanceLoader, EDGARLoader, FREDLoader, LendingClubLoader
    REAL_WORLD_AVAILABLE = True
except ImportError:
    REAL_WORLD_AVAILABLE = False
    print("⚠️ Real-world data loaders not available. Install requirements_real_world.txt")

def load_enhanced_data(source: str = "sample", **kwargs) -> pd.DataFrame:
    """
    Load data from various sources.
    
    Args:
        source: Data source ('sample', 'yahoo', 'edgar', 'lendingclub', 'fred')
        **kwargs: Additional parameters for specific sources
    
    Returns:
        DataFrame with financial data
    """
    
    if source == "sample":
        # Load original sample data
        sample_path = Path("data/sample_financials.csv")
        if sample_path.exists():
            df = pd.read_csv(sample_path)
        else:
            raise FileNotFoundError("Sample data file not found")
    
    elif source in ["yahoo", "edgar", "lendingclub", "fred"] and REAL_WORLD_AVAILABLE:
        from integrate_real_data import load_real_world_data
        df = load_real_world_data(source, **kwargs)
    
    else:
        raise ValueError(f"Unknown or unavailable data source: {source}")
    
    return df


def validate_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate loaded data for quality and completeness.
    """
    results = {"valid": True, "warnings": [], "errors": []}
    
    # Check required columns
    required_cols = [
        'company_id', 'as_of_date', 'industry', 'region', 'revenue', 
        'ebitda', 'debt', 'cash', 'working_capital'
    ]
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        results["errors"].append(f"Missing required columns: {missing_cols}")
        results["valid"] = False
    
    # Check for negative values where inappropriate
    numeric_cols = ['revenue', 'ebitda', 'debt', 'cash', 'working_capital']
    for col in numeric_cols:
        if col in df.columns:
            negative_count = (df[col] < 0).sum()
            if negative_count > 0:
                results["warnings"].append(f"{negative_count} records have negative {col}")
    
    # Check for extreme outliers
    if 'revenue' in df.columns:
        q99 = df['revenue'].quantile(0.99)
        outliers = (df['revenue'] > q99 * 10).sum()
        if outliers > 0:
            results["warnings"].append(f"{outliers} records have extremely high revenue")
    
    return results


def get_data_source_info():
    """
    Get information about available data sources.
    """
    sources = {
        "sample": {
            "description": "Original sample financial data",
            "size": "20 companies",
            "update_freq": "Static",
            "cost": "Free"
        }
    }
    
    if REAL_WORLD_AVAILABLE:
        sources.update({
            "yahoo": {
                "description": "Yahoo Finance market data and company info",
                "size": "Global stocks and indices",
                "update_freq": "Real-time",
                "cost": "Free"
            },
            "edgar": {
                "description": "SEC EDGAR financial filings",
                "size": "All US public companies",
                "update_freq": "Quarterly/Annual",
                "cost": "Free"
            },
            "lendingclub": {
                "description": "LendingClub loan performance data",
                "size": "2M+ historical loans",
                "update_freq": "Historical",
                "cost": "Free (Kaggle)"
            },
            "fred": {
                "description": "Federal Reserve economic data",
                "size": "800K+ economic series",
                "update_freq": "Daily/Monthly",
                "cost": "Free (API key required)"
            }
        })
    
    return sources


# Keep original functions for backward compatibility
def load_data(file_path: Optional[str] = None) -> pd.DataFrame:
    """Original load_data function for backward compatibility."""
    return load_enhanced_data("sample")
