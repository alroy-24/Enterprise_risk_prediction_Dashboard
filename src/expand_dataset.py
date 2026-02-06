"""
Script to expand the real-world dataset with more companies and data sources.
"""

import pandas as pd
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from real_world_data_loaders import YahooFinanceLoader, EDGARLoader, LendingClubLoader
from integrate_real_data import load_real_world_data


def expand_yahoo_finance_dataset():
    """Expand dataset with more companies from Yahoo Finance."""
    
    print("🔄 Expanding Yahoo Finance dataset...")
    
    # More diverse companies across different sectors
    expanded_tickers = [
        # Technology
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'ADBE',
        # Finance
        'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'AXP', 'BLK',
        # Healthcare
        'JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'T', 'ABT', 'CVS',
        # Energy
        'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'HAL', 'BP', 'SHEL',
        # Consumer
        'WMT', 'COST', 'HD', 'MCD', 'NKE', 'KO', 'PEP', 'PG',
        # Industrial
        'CAT', 'GE', 'HON', 'UPS', 'BA', 'MMM', 'DE', 'RTX'
    ]
    
    loader = YahooFinanceLoader()
    all_data = []
    
    for ticker in expanded_tickers:
        try:
            print(f"📊 Processing {ticker}...")
            company_info = loader.get_company_info(ticker)
            
            if company_info and company_info.get('revenue', 0) > 0:
                # Create multiple records for different time periods
                for year_offset in range(3):  # 2024, 2025, 2026
                    record = {
                        'company_id': f'{ticker}_{2024 + year_offset}',
                        'as_of_date': f'{2024 + year_offset}-12-31',
                        'industry': company_info.get('industry', 'Unknown'),
                        'region': company_info.get('region', 'NA'),
                        'revenue': company_info.get('revenue', 0) * (1 + year_offset * 0.05),  # 5% growth
                        'ebitda': company_info.get('ebitda', 0) * (1 + year_offset * 0.03),
                        'debt': company_info.get('total_debt', 0),
                        'cash': company_info.get('total_cash', 0),
                        'working_capital': company_info.get('revenue', 0) * 0.15,
                        'compliance_incidents': 0,
                        'default_probability': max(0.01, min(0.15, company_info.get('debt_to_equity', 0.5) * 0.05)),
                        'operational_incidents': 0,
                        'risk_flag': 0,
                        'total_assets': company_info.get('revenue', 0) * 1.5,
                        'total_equity': company_info.get('revenue', 0) * 1.2,
                        'current_assets': company_info.get('total_cash', 0) * 2,
                        'current_liabilities': company_info.get('revenue', 0) * 0.3,
                        'interest_expense': company_info.get('total_debt', 0) * 0.04,
                        'accounts_receivable': company_info.get('revenue', 0) * 0.15,
                        'net_income': company_info.get('revenue', 0) * 0.1,
                        'retained_earnings': company_info.get('revenue', 0) * 0.5,
                        'symbol': ticker,
                        'company_name': company_info.get('company_name', ''),
                        'sector': company_info.get('sector', ''),
                        'market_cap': company_info.get('market_cap', 0),
                        'beta': company_info.get('beta', 1.0),
                        'pe_ratio': company_info.get('pe_ratio', 0),
                        'debt_to_equity': company_info.get('debt_to_equity', 0),
                        'return_on_equity': company_info.get('return_on_equity', 0),
                        'total_debt': company_info.get('total_debt', 0),
                        'total_cash': company_info.get('total_cash', 0),
                        'current_ratio': company_info.get('current_ratio', 0),
                    }
                    all_data.append(record)
            
        except Exception as e:
            print(f"⚠️ Error processing {ticker}: {e}")
    
    return pd.DataFrame(all_data)


def add_lendingclub_data():
    """Add LendingClub loan data for credit risk modeling."""
    
    print("🔄 Adding LendingClub data...")
    
    try:
        # Try to load LendingClub data
        loan_data = load_real_world_data("lendingclub", sample_size=5000)
        
        if not loan_data.empty:
            # Map to our format and add diversity
            loan_data['company_id'] = ['LOAN_' + str(i) for i in range(len(loan_data))]
            loan_data['as_of_date'] = '2024-12-31'
            loan_data['industry'] = 'Consumer Finance'
            loan_data['region'] = 'NA'
            
            # Add some variation in risk flags based on interest rates
            loan_data['risk_flag'] = (loan_data['default_probability'] > 0.1).astype(int)
            
            return loan_data
        else:
            print("⚠️ LendingClub data not available")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"⚠️ Error loading LendingClub data: {e}")
        return pd.DataFrame()


def create_synthetic_variations(base_df: pd.DataFrame, n_variations: int = 1000):
    """Create synthetic variations to increase dataset size."""
    
    print(f"🔄 Creating {n_variations} synthetic variations...")
    
    if base_df.empty:
        return pd.DataFrame()
    
    synthetic_data = []
    
    for i in range(n_variations):
        # Randomly select a base record
        base_record = base_df.sample(1).iloc[0].copy()
        
        # Add realistic variations
        variation_factor = 0.8 + (i % 40) / 100  # 0.8 to 1.2
        
        # Vary financial metrics
        base_record['revenue'] *= variation_factor
        base_record['ebitda'] *= variation_factor
        base_record['debt'] *= (0.9 + (i % 30) / 100)
        base_record['cash'] *= (0.85 + (i % 35) / 100)
        
        # Vary risk metrics
        base_record['default_probability'] = max(0.01, min(0.2, 
            base_record['default_probability'] * (0.8 + (i % 50) / 100)))
        base_record['operational_incidents'] = i % 5
        base_record['compliance_incidents'] = i % 3
        
        # Update derived fields
        base_record['working_capital'] = base_record['current_assets'] - base_record['current_liabilities']
        base_record['total_assets'] = base_record['revenue'] * 1.5
        base_record['total_equity'] = base_record['total_assets'] - base_record['debt']
        
        # Create unique ID
        base_record['company_id'] = f"SYNTH_{i:04d}"
        
        synthetic_data.append(base_record)
    
    return pd.DataFrame(synthetic_data)


def main():
    """Main function to create expanded dataset."""
    
    print("🚀 Creating Expanded Real-World Dataset")
    print("=" * 50)
    
    # Step 1: Expand Yahoo Finance data
    yahoo_df = expand_yahoo_finance_dataset()
    print(f"✅ Yahoo Finance: {len(yahoo_df)} records")
    
    # Step 2: Add LendingClub data
    loan_df = add_lendingclub_data()
    print(f"✅ LendingClub: {len(loan_df)} records")
    
    # Step 3: Create synthetic variations
    combined_df = pd.concat([yahoo_df, loan_df], ignore_index=True) if not loan_df.empty else yahoo_df
    synthetic_df = create_synthetic_variations(combined_df, n_variations=2000)
    print(f"✅ Synthetic: {len(synthetic_df)} records")
    
    # Step 4: Combine all data
    final_df = pd.concat([combined_df, synthetic_df], ignore_index=True)
    
    # Step 5: Add feature engineering
    from features import engineer_features
    final_df = engineer_features(final_df)
    
    # Step 6: Save expanded dataset
    output_path = "data/expanded_real_world_financials.csv"
    final_df.to_csv(output_path, index=False)
    
    print(f"\n📈 Expanded Dataset Summary:")
    print(f"📊 Total records: {len(final_df)}")
    print(f"🏢 Unique companies: {final_df['company_id'].nunique()}")
    print(f"🏭 Industries: {final_df['industry'].nunique()}")
    print(f"🌍 Regions: {final_df['region'].nunique()}")
    print(f"💾 Saved to: {output_path}")
    
    # Show sample
    print(f"\n📋 Sample records:")
    print(final_df[['company_id', 'industry', 'revenue', 'debt', 'default_probability']].head(10))
    
    return final_df


if __name__ == "__main__":
    expanded_data = main()
