"""
Integration script to incorporate real-world data into the existing risk platform.
This script modifies the data_ingest.py to support real-world data sources.
"""

import pandas as pd
import sys
import os
import yaml
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from real_world_data_loaders import YahooFinanceLoader, EDGARLoader, FREDLoader, LendingClubLoader
from data_ingest import load_data, validate_columns
from features import engineer_features


def load_real_world_data(source: str = "yahoo", **kwargs) -> pd.DataFrame:
    """
    Load real-world data from various sources and return in platform format.
    
    Args:
        source: Data source ('yahoo', 'edgar', 'fred', 'lendingclub', 'sample')
        **kwargs: Additional parameters for specific loaders
    
    Returns:
        DataFrame in the same format as sample_financials.csv
    """
    
    if source == "yahoo":
        loader = YahooFinanceLoader()
        tickers = kwargs.get('tickers', ['AAPL', 'MSFT', 'JPM', 'XOM', 'WMT'])
        
        # Get company info for each ticker
        all_data = []
        for ticker in tickers:
            company_info = loader.get_company_info(ticker)
            if company_info:
                # Create a single record for each company
                record = {
                    'company_id': ticker,
                    'as_of_date': '2024-12-31',
                    'industry': company_info.get('industry', 'Unknown'),
                    'region': company_info.get('region', 'NA'),
                    'revenue': company_info.get('revenue', 0),
                    'ebitda': company_info.get('ebitda', 0),
                    'debt': company_info.get('total_debt', 0),
                    'cash': company_info.get('total_cash', 0),
                    'working_capital': company_info.get('revenue', 0) * 0.15,  # Estimate
                    'compliance_incidents': 0,
                    'default_probability': 0.05,  # Default estimate
                    'operational_incidents': 0,
                    'risk_flag': 0,
                    'total_assets': company_info.get('revenue', 0) * 1.5,  # Estimate
                    'total_equity': company_info.get('revenue', 0) * 1.2,  # Estimate
                    'current_assets': company_info.get('total_cash', 0) * 2,  # Estimate
                    'current_liabilities': company_info.get('revenue', 0) * 0.3,  # Estimate
                    'interest_expense': company_info.get('total_debt', 0) * 0.04,  # 4% interest
                    'accounts_receivable': company_info.get('revenue', 0) * 0.15,  # 15% of revenue
                    'net_income': company_info.get('revenue', 0) * 0.1,  # 10% margin
                    'retained_earnings': company_info.get('revenue', 0) * 0.5,  # Estimate
                }
                all_data.append(record)
        
        df = pd.DataFrame(all_data)
        
    elif source == "edgar":
        loader = EDGARLoader()
        ciks = kwargs.get('ciks', ['0000320193', '0000789019', '0000019617'])  # Apple, Microsoft, JPMorgan
        years = kwargs.get('years', 3)
        
        all_data = []
        for cik in ciks:
            company_data = loader.get_company_financials(cik, years)
            if not company_data.empty:
                all_data.append(company_data)
        
        df = pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()
        
    elif source == "lendingclub":
        loader = LendingClubLoader(kwargs.get('data_path'))
        sample_size = kwargs.get('sample_size', 5000)
        df = loader.load_loan_data(sample_size)
        
    elif source == "sample":
        # Create sample dataset using multiple sources
        from real_world_data_loaders import create_sample_real_world_dataset
        df = create_sample_real_world_dataset()
        
    else:
        raise ValueError(f"Unknown data source: {source}")
    
    if df.empty:
        print(f"❌ No data loaded from source: {source}")
        return pd.DataFrame()
    
    # Validate and engineer features
    print(f"🔍 Validating {len(df)} records...")
    try:
        df = validate_columns(df)
        print("✅ Data validation passed")
    except ValueError as e:
        print(f"⚠️ Data validation warnings: {e}")
    
    df = engineer_features(df)
    print("✅ Feature engineering completed")
    
    return df


def create_enhanced_data_ingest():
    """
    Create an enhanced version of data_ingest.py that supports real-world data sources.
    """
    
    enhanced_code = '''"""
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
'''
    
    # Write the enhanced data ingestion module
    with open('src/data_ingest_enhanced.py', 'w', encoding='utf-8') as f:
        f.write(enhanced_code)
    
    print("✅ Created enhanced data ingestion module: src/data_ingest_enhanced.py")


def update_config_for_real_data():
    """
    Update configuration files to support real-world data sources.
    """
    
    config_updates = {
        'data_sources': {
            'default': 'sample',
            'available': ['sample', 'yahoo', 'edgar', 'lendingclub', 'fred'],
            'yahoo': {
                'default_tickers': ['AAPL', 'MSFT', 'JPM', 'XOM', 'WMT', 'GOOGL', 'AMZN', 'META'],
                'period': '5y'
            },
            'edgar': {
                'default_ciks': ['0000320193', '0000789019', '0000019617', '0000051143', '0000104169'],
                'years': 3
            },
            'lendingclub': {
                'data_path': 'data/lendingclub_loan_data.csv',
                'sample_size': 10000
            },
            'fred': {
                'api_key': None,  # User needs to provide
                'default_series': ['GDP', 'UNRATE', 'FEDFUNDS', 'DGS10']
            }
        }
    }
    
    # Save as new config file
    with open('config/real_world_config.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(config_updates, f, default_flow_style=False)
    
    print("✅ Created real-world data configuration: config/real_world_config.yaml")


def main():
    """
    Main function to set up real-world data integration.
    """
    
    print("🚀 Setting up Real-World Data Integration")
    print("=" * 50)
    
    # Step 1: Create enhanced data ingestion
    create_enhanced_data_ingest()
    
    # Step 2: Update configuration
    update_config_for_real_data()
    
    # Step 3: Test with sample real data
    print("\n📊 Testing real-world data loading...")
    
    try:
        # Test Yahoo Finance data
        yahoo_data = load_real_world_data("yahoo", tickers=['AAPL', 'MSFT'])
        if not yahoo_data.empty:
            print(f"✅ Yahoo Finance: {len(yahoo_data)} companies loaded")
        
        # Test sample dataset creation
        sample_data = load_real_world_data("sample")
        if not sample_data.empty:
            print(f"✅ Sample dataset: {len(sample_data)} records loaded")
        
    except Exception as e:
        print(f"⚠️ Error during testing: {e}")
    
    print("\n📋 Setup Complete!")
    print("\n💡 Next Steps:")
    print("1. Install additional dependencies: pip install -r requirements_real_world.txt")
    print("2. For FRED data, get API key from: https://fred.stlouisfed.org/docs/api/api_key.html")
    print("3. For LendingClub data, download from: https://www.kaggle.com/datasets/wordsforthewise/lending-club")
    print("4. Update your app.py to use: from data_ingest_enhanced import load_enhanced_data")
    print("5. Test with: df = load_enhanced_data('yahoo', tickers=['AAPL', 'MSFT', 'GOOGL'])")


if __name__ == "__main__":
    main()
