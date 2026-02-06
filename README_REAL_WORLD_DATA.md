# Real-World Data Integration Guide

This guide explains how to integrate real-world financial data into your Enterprise Risk Intelligence Platform.

## 🚀 Quick Start

### 1. Install Additional Dependencies

```bash
pip install -r requirements_real_world.txt
```

### 2. Test Real-World Data Loading

```python
from src.integrate_real_data import load_real_world_data

# Load Yahoo Finance data
df = load_real_world_data("yahoo", tickers=['AAPL', 'MSFT', 'JPM'])
print(f"Loaded {len(df)} companies")

# Load EDGAR financial data
df_edgar = load_real_world_data("edgar", ciks=['0000320193', '0000789019'])
print(f"Loaded {len(df_edgar)} financial records")
```

## 📊 Available Data Sources

### 1. Yahoo Finance (Free, Real-time)
- **Coverage**: Global stocks, indices, ETFs
- **Data**: Market prices, company fundamentals, key ratios
- **Use Case**: Market risk, VaR calculations, real-time monitoring
- **API**: `yfinance` Python package

```python
from src.real_world_data_loaders import YahooFinanceLoader

loader = YahooFinanceLoader()
market_data = loader.get_market_data(['AAPL', 'MSFT', 'GOOGL'], period='5y')
company_info = loader.get_company_info('AAPL')
```

### 2. SEC EDGAR (Free, Official)
- **Coverage**: All US public companies
- **Data**: Financial statements, 10-K/10-Q filings
- **Use Case**: Fundamental analysis, credit risk, compliance
- **API**: SEC Data API

```python
from src.real_world_data_loaders import EDGARLoader

loader = EDGARLoader()
financials = loader.get_company_financials('0000320193', years=5)  # Apple
```

### 3. Federal Reserve FRED (Free, API Key Required)
- **Coverage**: 800K+ economic indicators
- **Data**: GDP, unemployment, interest rates, inflation
- **Use Case**: Stress testing, macro scenarios, market risk
- **API Key**: Get from [FRED](https://fred.stlouisfed.org/docs/api/api_key.html)

```python
from src.real_world_data_loaders import FREDLoader

loader = FREDLoader(api_key='your_api_key')
macro_data = loader.get_macro_indicators(['GDP', 'UNRATE', 'FEDFUNDS'])
```

### 4. LendingClub (Free via Kaggle)
- **Coverage**: 2M+ peer-to-peer loans
- **Data**: Loan performance, defaults, credit characteristics
- **Use Case**: Credit risk modeling, default prediction
- **Download**: [Kaggle Dataset](https://www.kaggle.com/datasets/wordsforthewise/lending-club)

```python
from src.real_world_data_loaders import LendingClubLoader

loader = LendingClubLoader("data/lendingclub_loan_data.csv")
loan_data = loader.load_loan_data(sample_size=10000)
```

## 🔧 Integration with Existing Platform

### Update Data Loading in Your App

Replace the existing data loading in `app.py`:

```python
# Original
from src.data_ingest import load_data
df = load_data()

# Enhanced version
from src.integrate_real_data import load_real_world_data

# Load different data sources
df_yahoo = load_real_world_data("yahoo", tickers=['AAPL', 'MSFT', 'JPM'])
df_edgar = load_real_world_data("edgar", ciks=['0000320193', '0000789019'])
df_loans = load_real_world_data("lendingclub", sample_size=5000)

# Combine datasets
df_combined = pd.concat([df_yahoo, df_edgar, df_loans], ignore_index=True)
```

### Configuration Options

Create `config/real_world_config.yaml`:

```yaml
data_sources:
  default: "yahoo"
  
  yahoo:
    default_tickers: ['AAPL', 'MSFT', 'JPM', 'XOM', 'WMT']
    period: '5y'
    
  edgar:
    default_ciks: ['0000320193', '0000789019', '0000019617']
    years: 3
    
  fred:
    api_key: null  # Set your API key here
    default_series: ['GDP', 'UNRATE', 'FEDFUNDS', 'DGS10']
    
  lendingclub:
    data_path: 'data/lendingclub_loan_data.csv'
    sample_size: 10000
```

## 📈 Use Case Examples

### 1. Enhanced Market Risk Analysis

```python
# Load market data for VaR calculations
yahoo_loader = YahooFinanceLoader()
market_data = yahoo_loader.get_market_data(['AAPL', 'MSFT', 'JPM'], period='2y')

# Calculate portfolio VaR using existing var_calculations.py
from src.var_calculations import calculate_var

portfolio_returns = market_data.mean(axis=1)
var_95 = calculate_var(portfolio_returns, confidence=0.95, method='historical')
```

### 2. Credit Risk with Real Loan Data

```python
# Load LendingClub data for credit risk modeling
loan_data = load_real_world_data("lendingclub", sample_size=20000)

# Use existing models.py for training
from src.models import train_risk_model

X, y = prepare_features(loan_data)
model = train_risk_model(X, y)
```

### 3. Stress Testing with Macro Data

```python
# Load FRED macro indicators for stress scenarios
fred_loader = FREDLoader(api_key='your_key')
macro_data = fred_loader.get_macro_indicators(['GDP', 'UNRATE', 'FEDFUNDS'])

# Use existing scenarios.py for stress testing
from src.scenarios import apply_stress_scenario

stressed_data = apply_stress_scenario(financial_data, macro_scenarios)
```

## 🔄 Data Source Mapping

| Platform Field | Yahoo Finance | EDGAR | LendingClub |
|---------------|---------------|-------|-------------|
| `revenue` | `totalRevenue` | `Revenues` | `annual_inc` |
| `ebitda` | `ebitda` | `OperatingIncomeLoss` | `annual_inc` * 0.2 |
| `debt` | `totalDebt` | `LongTermDebt` | `loan_amnt` |
| `cash` | `totalCash` | `CashAndCashEquivalents` | `annual_inc` * 0.1 |
| `default_probability` | N/A | N/A | `int_rate` / 100 |
| `industry` | `industry` | `category` | "Consumer Finance" |
| `region` | Mapped from country | Mapped from state | "NA" |

## 🛡️ Rate Limiting & Best Practices

### Yahoo Finance
- **Rate Limit**: ~2000 requests/hour
- **Best Practice**: Cache data, use bulk downloads

### SEC EDGAR
- **Rate Limit**: 10 requests/second
- **Best Practice**: Include User-Agent, respect robots.txt

### FRED
- **Rate Limit**: 120 requests/minute
- **Best Practice**: Batch requests, use API key

## 🚨 Important Notes

1. **API Keys**: Only FRED requires an API key (free registration)
2. **Data Quality**: Real-world data may have missing values or inconsistencies
3. **Validation**: Always validate data before using in risk models
4. **Caching**: Consider caching API responses to improve performance
5. **Compliance**: Ensure data usage complies with terms of service

## 🔍 Troubleshooting

### Common Issues

1. **Import Errors**: Install requirements_real_world.txt
2. **API Limits**: Add delays between requests
3. **Missing Data**: Check data availability for specific companies/periods
4. **Format Issues**: Data may need cleaning before feature engineering

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# This will show detailed API request information
df = load_real_world_data("yahoo", tickers=['AAPL'])
```

## 📚 Additional Resources

- [Yahoo Finance API Documentation](https://pypi.org/project/yfinance/)
- [SEC EDGAR API Guide](https://www.sec.gov/edgar/sec-api-documentation)
- [FRED API Documentation](https://fred.stlouisfed.org/docs/api/fred/)
- [LendingClub Data Dictionary](https://www.lendingclub.com/info/dataset.action)

## 🎯 Next Steps

1. **Test Integration**: Start with Yahoo Finance (easiest)
2. **Add EDGAR**: For detailed financial statements
3. **Configure FRED**: For macro stress testing
4. **Deploy**: Update your Streamlit app to use real data
5. **Monitor**: Set up data quality checks and alerts

---

**Need help?** Check the example notebooks in `examples/` or open an issue on GitHub.
