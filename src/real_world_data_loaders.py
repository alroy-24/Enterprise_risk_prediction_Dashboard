"""
Real-world data loaders for Enterprise Risk Intelligence Platform.
Supports Yahoo Finance, EDGAR filings, FRED economic data, and LendingClub.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import time
import warnings
warnings.filterwarnings('ignore')

try:
    from fredapi import Fred
    FRED_AVAILABLE = True
except ImportError:
    FRED_AVAILABLE = False
    print("FRED not available. Install with: pip install fredapi")

class YahooFinanceLoader:
    """Load market data from Yahoo Finance for VaR and market risk analysis."""
    
    def __init__(self):
        self.session = requests.Session()
    
    def get_market_data(self, tickers: List[str], period: str = "5y") -> pd.DataFrame:
        """
        Download historical market data for VaR calculations.
        
        Args:
            tickers: List of stock symbols (e.g., ['AAPL', 'MSFT', 'JPM'])
            period: Time period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
        
        Returns:
            DataFrame with OHLCV data and returns
        """
        try:
            data = yf.download(tickers, period=period, group_by='ticker', progress=False)
            
            if len(tickers) == 1:
                data.columns = pd.MultiIndex.from_product([tickers, data.columns])
            
            # Calculate returns for each ticker
            returns_data = {}
            for ticker in tickers:
                if ticker in data.columns.get_level_values(0):
                    ticker_data = data[ticker]
                    returns_data[f'{ticker}_return'] = ticker_data['Adj Close'].pct_change()
                    returns_data[f'{ticker}_volatility'] = ticker_data['Adj Close'].rolling(20).std()
                    returns_data[f'{ticker}_volume'] = ticker_data['Volume']
            
            returns_df = pd.DataFrame(returns_data, index=data.index)
            
            print(f"✅ Downloaded market data for {len(tickers)} tickers ({len(data)} observations)")
            return returns_df.dropna()
            
        except Exception as e:
            print(f"❌ Error downloading market data: {e}")
            return pd.DataFrame()
    
    def get_company_info(self, ticker: str) -> Dict:
        """Get company information including sector, industry, and key metrics."""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            return {
                'symbol': ticker,
                'company_name': info.get('longName', ''),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'market_cap': info.get('marketCap', 0),
                'beta': info.get('beta', 1.0),
                'pe_ratio': info.get('trailingPE', 0),
                'debt_to_equity': info.get('debtToEquity', 0),
                'return_on_equity': info.get('returnOnEquity', 0),
                'revenue': info.get('totalRevenue', 0),
                'ebitda': info.get('ebitda', 0),
                'total_debt': info.get('totalDebt', 0),
                'total_cash': info.get('totalCash', 0),
                'current_ratio': info.get('currentRatio', 0),
                'region': self._get_region_from_country(info.get('country', 'US'))
            }
        except Exception as e:
            print(f"❌ Error getting company info for {ticker}: {e}")
            return {}
    
    def _get_region_from_country(self, country: str) -> str:
        """Map country to region for consistency with existing data."""
        country_region_map = {
            'United States': 'NA',
            'Canada': 'NA',
            'Mexico': 'NA',
            'United Kingdom': 'EU',
            'Germany': 'EU',
            'France': 'EU',
            'Italy': 'EU',
            'Spain': 'EU',
            'Netherlands': 'EU',
            'China': 'APAC',
            'Japan': 'APAC',
            'South Korea': 'APAC',
            'India': 'APAC',
            'Australia': 'APAC',
            'Singapore': 'APAC'
        }
        return country_region_map.get(country, 'Other')


class EDGARLoader:
    """Load financial data from SEC EDGAR filings."""
    
    def __init__(self):
        self.base_url = "https://data.sec.gov/api/xbrl"
        self.headers = {
            'User-Agent': 'Enterprise Risk Platform your.email@company.com',
            'Accept-Encoding': 'gzip, deflate',
            'Host': 'data.sec.gov'
        }
    
    def get_company_financials(self, cik: str, years: int = 3) -> pd.DataFrame:
        """
        Get financial data from SEC EDGAR for a company.
        
        Args:
            cik: Central Index Key (10-digit CIK number)
            years: Number of years of data to fetch
        
        Returns:
            DataFrame with financial metrics in the same format as sample_financials.csv
        """
        try:
            # Get company facts
            facts_url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik.zfill(10)}.json"
            response = requests.get(facts_url, headers=self.headers)
            response.raise_for_status()
            
            data = response.json()
            entity_data = data['facts']['us-gaap']
            
            # Get the most recent fiscal years
            end_date = datetime.now()
            start_date = end_date - timedelta(days=years * 365)
            
            financial_data = []
            
            # Extract key financial metrics
            metrics_map = {
                'Revenues': 'revenue',
                'OperatingIncomeLoss': 'ebitda',  # Approximation
                'LongTermDebt': 'debt',
                'CashAndCashEquivalentsAtCarryingValue': 'cash',
                'AssetsCurrent': 'current_assets',
                'LiabilitiesCurrent': 'current_liabilities',
                'Assets': 'total_assets',
                'StockholdersEquity': 'total_equity',
                'InterestExpense': 'interest_expense',
                'AccountsReceivableNetCurrent': 'accounts_receivable',
                'NetIncomeLoss': 'net_income',
                'RetainedEarningsAccumulatedDeficit': 'retained_earnings'
            }
            
            for year_offset in range(years):
                year_data = {'company_id': f'EDGAR{cik}', 'as_of_date': f'{end_date.year - year_offset}-12-31'}
                
                for sec_metric, our_metric in metrics_map.items():
                    if sec_metric in entity_data:
                        units_data = entity_data[sec_metric]['units']['USD']
                        # Find the value closest to our target date
                        target_date = end_date - timedelta(days=year_offset * 365)
                        
                        closest_value = None
                        min_diff = float('inf')
                        
                        for item in units_data:
                            item_date = datetime.strptime(item['end'], '%Y-%m-%d')
                            diff = abs((item_date - target_date).days)
                            if diff < min_diff and diff < 180:  # Within 6 months
                                min_diff = diff
                                closest_value = item['val']
                        
                        year_data[our_metric] = closest_value or 0
                    else:
                        year_data[our_metric] = 0
                
                financial_data.append(year_data)
            
            df = pd.DataFrame(financial_data)
            
            # Add company metadata
            company_info = self._get_company_info(cik)
            for key, value in company_info.items():
                df[key] = value
            
            # Calculate derived fields
            df['working_capital'] = df['current_assets'] - df['current_liabilities']
            df['compliance_incidents'] = 0  # Not available in EDGAR
            df['operational_incidents'] = 0  # Not available in EDGAR
            df['default_probability'] = 0.05  # Default estimate
            df['risk_flag'] = 0
            
            print(f"✅ Loaded EDGAR data for CIK {cik} ({len(df)} years)")
            return df
            
        except Exception as e:
            print(f"❌ Error loading EDGAR data for CIK {cik}: {e}")
            return pd.DataFrame()
    
    def _get_company_info(self, cik: str) -> Dict:
        """Get basic company information from SEC."""
        try:
            url = f"https://data.sec.gov/submissions/CIK{cik.zfill(10)}.json"
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            
            data = response.json()
            company_info = data.get('entity', {})
            
            return {
                'industry': company_info.get('category', 'Unknown'),
                'region': self._get_region_from_state(company_info.get('stateOfIncorporation', ''))
            }
        except:
            return {'industry': 'Unknown', 'region': 'Other'}
    
    def _get_region_from_state(self, state: str) -> str:
        """Map US state to region."""
        na_states = {'CA', 'OR', 'WA', 'NV', 'AZ', 'AK', 'HI', 'NY', 'NJ', 'CT', 'MA', 'RI', 'VT', 'NH', 'ME', 'PA', 'TX', 'OK', 'AR', 'LA', 'FL', 'GA', 'AL', 'MS', 'SC', 'NC', 'TN', 'KY', 'WV', 'VA'}
        return 'NA' if state in na_states else 'Other'


class FREDLoader:
    """Load macroeconomic data from Federal Reserve Economic Data."""
    
    def __init__(self, api_key: Optional[str] = None):
        if FRED_AVAILABLE and api_key:
            self.fred = Fred(api_key=api_key)
        else:
            self.fred = None
            print("⚠️ FRED loader requires API key. Get one at: https://fred.stlouisfed.org/docs/api/api_key.html")
    
    def get_macro_indicators(self, series_ids: List[str], start_date: str = "2020-01-01") -> pd.DataFrame:
        """
        Get macroeconomic indicators for stress testing scenarios.
        
        Args:
            series_ids: List of FRED series IDs (e.g., ['GDP', 'UNRATE', 'FEDFUNDS'])
            start_date: Start date for data
        
        Returns:
            DataFrame with macro indicators
        """
        if not self.fred:
            print("❌ FRED not initialized with API key")
            return pd.DataFrame()
        
        try:
            macro_data = {}
            
            for series_id in series_ids:
                try:
                    data = self.fred.get_series(series_id, observation_start=start_date)
                    macro_data[series_id] = data
                    time.sleep(0.1)  # Rate limiting
                except Exception as e:
                    print(f"⚠️ Could not fetch {series_id}: {e}")
            
            df = pd.DataFrame(macro_data)
            print(f"✅ Loaded {len(df)} observations for {len(macro_data)} FRED series")
            return df
            
        except Exception as e:
            print(f"❌ Error loading FRED data: {e}")
            return pd.DataFrame()
    
    def get_risk_free_rates(self) -> pd.DataFrame:
        """Get risk-free rates for VaR calculations."""
        if not self.fred:
            return pd.DataFrame()
        
        risk_free_series = [
            'DGS10',  # 10-Year Treasury Constant Maturity Rate
            'DGS2',   # 2-Year Treasury Constant Maturity Rate
            'DGS3MO',  # 3-Month Treasury Constant Maturity Rate
            'DFF',    # Federal Funds Effective Rate
        ]
        
        return self.get_macro_indicators(risk_free_series)


class LendingClubLoader:
    """Load loan data from LendingClub for credit risk modeling."""
    
    def __init__(self, data_path: str = None):
        self.data_path = data_path or "data/lendingclub_loan_data.csv"
    
    def load_loan_data(self, sample_size: int = 10000) -> pd.DataFrame:
        """
        Load LendingClub loan data and map to risk platform format.
        
        Args:
            sample_size: Number of loans to sample (for memory efficiency)
        
        Returns:
            DataFrame mapped to risk platform schema
        """
        try:
            # Try to load from local file first
            try:
                df = pd.read_csv(self.data_path, nrows=sample_size)
            except FileNotFoundError:
                print("⚠️ LendingClub data not found locally. Download from:")
                print("https://www.kaggle.com/datasets/wordsforthewise/lending-club")
                return pd.DataFrame()
            
            # Map LendingClub fields to our schema
            mapped_data = {
                'company_id': ['LC' + str(i) for i in range(len(df))],
                'as_of_date': ['2024-12-31'] * len(df),
                'industry': ['Consumer Finance'] * len(df),
                'region': ['NA'] * len(df),
                'revenue': df['annual_inc'].fillna(50000),  # Annual income as proxy
                'ebitda': df['annual_inc'].fillna(50000) * 0.2,  # Estimate
                'debt': df['loan_amnt'].fillna(10000),
                'cash': df['annual_inc'].fillna(50000) * 0.1,  # Estimate
                'working_capital': df['annual_inc'].fillna(50000) * 0.15,  # Estimate
                'compliance_incidents': 0,
                'default_probability': df['int_rate'].fillna(10) / 100,  # Interest rate as proxy
                'operational_incidents': df['delinq_2yrs'].fillna(0),
                'risk_flag': (df['loan_status'] == 'Charged Off').astype(int),
                'total_assets': df['annual_inc'].fillna(50000) * 2,  # Estimate
                'total_equity': df['annual_inc'].fillna(50000) * 1.5,  # Estimate
                'current_assets': df['annual_inc'].fillna(50000) * 0.3,  # Estimate
                'current_liabilities': df['annual_inc'].fillna(50000) * 0.15,  # Estimate
                'interest_expense': df['loan_amnt'].fillna(10000) * df['int_rate'].fillna(10) / 100,
                'accounts_receivable': 0,  # Not applicable for individuals
                'net_income': df['annual_inc'].fillna(50000) * 0.1,  # Estimate
                'retained_earnings': df['annual_inc'].fillna(50000) * 0.5,  # Estimate
            }
            
            result_df = pd.DataFrame(mapped_data)
            print(f"✅ Loaded {len(result_df)} LendingClub records")
            return result_df
            
        except Exception as e:
            print(f"❌ Error loading LendingClub data: {e}")
            return pd.DataFrame()


def create_sample_real_world_dataset():
    """Create a sample dataset using real-world data sources."""
    
    print("🔄 Creating sample real-world dataset...")
    
    # Initialize loaders
    yahoo_loader = YahooFinanceLoader()
    edgar_loader = EDGARLoader()
    
    # Get data for major companies (CIKs for Apple, Microsoft, JPMorgan)
    sample_companies = [
        {'ticker': 'AAPL', 'cik': '0000320193'},
        {'ticker': 'MSFT', 'cik': '0000789019'},
        {'ticker': 'JPM', 'cik': '0000019617'},
        {'ticker': 'XOM', 'cik': '0000051143'},
        {'ticker': 'WMT', 'cik': '0000104169'}
    ]
    
    all_data = []
    
    for company in sample_companies:
        print(f"\n📊 Processing {company['ticker']}...")
        
        # Get company info from Yahoo Finance
        company_info = yahoo_loader.get_company_info(company['ticker'])
        
        # Get financial data from EDGAR
        financial_data = edgar_loader.get_company_financials(company['cik'], years=3)
        
        if not financial_data.empty:
            # Merge company info with financial data
            for key, value in company_info.items():
                if key not in ['revenue', 'ebitda', 'debt', 'cash']:  # Don't overwrite financial data
                    financial_data[key] = value
            
            all_data.append(financial_data)
    
    # Combine all data
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Save to data directory
        output_path = "data/real_world_financials.csv"
        combined_df.to_csv(output_path, index=False)
        print(f"\n✅ Real-world dataset saved to {output_path}")
        print(f"📈 Dataset contains {len(combined_df)} records across {combined_df['company_id'].nunique()} companies")
        
        return combined_df
    else:
        print("❌ No data could be loaded")
        return pd.DataFrame()


if __name__ == "__main__":
    # Example usage
    print("🚀 Real-World Data Loaders for Enterprise Risk Platform")
    print("=" * 60)
    
    # Create sample dataset
    sample_data = create_sample_real_world_dataset()
    
    if not sample_data.empty:
        print("\n📋 Sample of loaded data:")
        print(sample_data[['company_id', 'as_of_date', 'industry', 'region', 'revenue', 'debt']].head())
        
        print("\n💡 To use in your main application:")
        print("from src.real_world_data_loaders import YahooFinanceLoader, EDGARLoader")
        print("loader = YahooFinanceLoader()")
        print("market_data = loader.get_market_data(['AAPL', 'MSFT', 'JPM'])")
