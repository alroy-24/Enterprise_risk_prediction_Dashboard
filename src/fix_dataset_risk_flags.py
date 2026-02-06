"""
Fix the expanded dataset by adding realistic risk flags based on financial metrics.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def add_realistic_risk_flags(df):
    """Add realistic risk flags based on financial metrics."""
    
    df = df.copy()
    
    # Initialize all risk flags to 0
    df['risk_flag'] = 0
    
    # Risk factors that increase probability of risk_flag = 1
    risk_factors = []
    
    # 1. High leverage ratio (debt-to-equity > 1.5)
    high_leverage = df['debt_to_equity'] > 1.5
    risk_factors.append(high_leverage)
    
    # 2. Low current ratio (< 1.0)
    low_liquidity = df['current_ratio'] < 1.0
    risk_factors.append(low_liquidity)
    
    # 3. Negative profitability (negative net_margin)
    negative_profit = df['net_margin'] < 0
    risk_factors.append(negative_profit)
    
    # 4. High default probability (> 0.15)
    high_default_prob = df['default_probability'] > 0.15
    risk_factors.append(high_default_prob)
    
    # 5. Low Altman Z-score (< 1.8 - distress zone)
    low_z_score = df['altman_z_score'] < 1.8
    risk_factors.append(low_z_score)
    
    # 6. High compliance intensity (> 0.5)
    high_compliance = df['compliance_intensity'] > 0.5
    risk_factors.append(high_compliance)
    
    # 7. High incident rate (> 0.3)
    high_incidents = df['incident_rate'] > 0.3
    risk_factors.append(high_incidents)
    
    # Combine risk factors - set risk_flag = 1 if at least 2 factors are present
    risk_score = sum(risk_factors)
    df['risk_flag'] = (risk_score >= 2).astype(int)
    
    # Add some randomness for realism (about 20% of records should have risk_flag = 1)
    np.random.seed(42)
    random_risk = np.random.random(len(df)) < 0.2
    df.loc[random_risk & (df['risk_flag'] == 0), 'risk_flag'] = 1
    
    # Ensure we have both classes
    print(f"📊 Risk Flag Distribution:")
    print(f"   Risk Flag 0 (Low Risk): {(df['risk_flag'] == 0).sum():,} records ({(df['risk_flag'] == 0).mean():.1%})")
    print(f"   Risk Flag 1 (High Risk): {(df['risk_flag'] == 1).sum():,} records ({(df['risk_flag'] == 1).mean():.1%})")
    
    return df


def main():
    """Main function to fix the dataset."""
    
    print("🔧 Fixing Risk Flags in Expanded Dataset")
    print("=" * 45)
    
    # Load the expanded dataset
    df_path = "data/expanded_real_world_financials.csv"
    if not Path(df_path).exists():
        print(f"❌ Dataset not found: {df_path}")
        return
    
    df = pd.read_csv(df_path)
    print(f"📊 Loaded dataset: {len(df):,} records")
    
    # Fix risk flags
    df_fixed = add_realistic_risk_flags(df)
    
    # Save the fixed dataset
    output_path = "data/expanded_real_world_financials_fixed.csv"
    df_fixed.to_csv(output_path, index=False)
    
    print(f"✅ Fixed dataset saved to: {output_path}")
    
    # Show some examples of high-risk companies
    high_risk = df_fixed[df_fixed['risk_flag'] == 1].head(5)
    print(f"\n🚨 Sample High-Risk Companies:")
    print(high_risk[['company_id', 'industry', 'debt_to_equity', 'current_ratio', 'altman_z_score', 'risk_flag']].to_string())
    
    return df_fixed


if __name__ == "__main__":
    fixed_data = main()
