"""
Fix market data to use real companies from expanded dataset.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def generate_market_data_real_companies(df_scored, start_date="2024-01-01", end_date="2024-12-31"):
    """Generate market data using real companies from dataset."""
    
    # Get unique companies from dataset
    companies = df_scored['company_id'].unique()
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    
    data = []
    np.random.seed(42)
    
    for company in companies:
        # Get company-specific data for realistic parameters
        company_data = df_scored[df_scored['company_id'] == company].iloc[0]
        
        # Base price based on company size (revenue)
        revenue = company_data.get('revenue', 1e9)
        base_price = np.clip(np.log10(revenue) * 10, 50, 500)
        
        # Generate prices
        prices = [base_price]
        returns = np.random.normal(0.0005, 0.02, len(dates) - 1)
        
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))
        
        # Yields based on company risk
        risk = company_data.get('predicted_proba', 0.5)
        base_yield = 0.02 + risk * 0.04  # Higher risk = higher yield
        yields = base_yield + np.random.normal(0, 0.001, len(dates))
        yields = np.clip(yields, 0.01, 0.10)
        
        # Credit spreads
        base_spread = 0.005 + risk * 0.03
        spreads = base_spread + np.random.normal(0, 0.002, len(dates))
        spreads = np.clip(spreads, 0, 0.10)
        
        # Volatility
        base_vol = 0.15 + risk * 0.20
        vols = base_vol + np.random.normal(0, 0.02, len(dates))
        vols = np.clip(vols, 0.10, 0.50)
        
        for i, date in enumerate(dates):
            data.append({
                "company_id": company,
                "date": date,
                "equity_price": prices[i],
                "yield": yields[i],
                "credit_spread": spreads[i],
                "volatility": vols[i]
            })
    
    return pd.DataFrame(data)


def main():
    """Update market data generation in app.py"""
    
    # Load the fixed dataset
    df = pd.read_csv("data/expanded_real_world_financials_fixed.csv")
    
    # Generate market data with real companies
    market_df = generate_market_data_real_companies(df)
    
    print(f"✅ Generated market data for {market_df['company_id'].nunique()} real companies")
    print(f"📊 Total records: {len(market_df)}")
    print(f"📅 Date range: {market_df['date'].min()} to {market_df['date'].max()}")
    
    # Save for reference
    market_df.to_csv("data/market_data_real_companies.csv", index=False)
    print("💾 Saved to data/market_data_real_companies.csv")
    
    return market_df


if __name__ == "__main__":
    main()
