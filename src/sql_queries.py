"""
SQL Queries Module
Industry-standard SQL queries for risk analytics.
Emphasizes SQL skills for professional data analysis.
"""
from typing import Optional
import pandas as pd
from sqlalchemy import create_engine, text


# SQL Views and Queries for Risk Analytics
RISK_SCORES_VIEW = """
CREATE OR REPLACE VIEW risk_scores_view AS
SELECT 
    company_id,
    industry,
    region,
    as_of_date,
    predicted_proba AS risk_probability,
    leverage_ratio,
    current_ratio,
    interest_coverage,
    ebitda_margin,
    altman_z_score,
    aggregated_risk,
    risk_rating
FROM risk_scored_data
WHERE as_of_date = (SELECT MAX(as_of_date) FROM risk_scored_data);
"""


HIGH_RISK_COMPANIES_QUERY = """
SELECT 
    company_id,
    industry,
    region,
    risk_probability,
    risk_rating,
    leverage_ratio,
    current_ratio
FROM risk_scores_view
WHERE risk_probability > 0.5
ORDER BY risk_probability DESC;
"""


INDUSTRY_BENCHMARKS_QUERY = """
SELECT 
    industry,
    COUNT(*) AS company_count,
    AVG(risk_probability) AS avg_risk_probability,
    AVG(leverage_ratio) AS avg_leverage_ratio,
    AVG(current_ratio) AS avg_current_ratio,
    AVG(ebitda_margin) AS avg_ebitda_margin,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY risk_probability) AS median_risk_probability,
    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY risk_probability) AS p25_risk_probability,
    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY risk_probability) AS p75_risk_probability
FROM risk_scores_view
GROUP BY industry
ORDER BY avg_risk_probability DESC;
"""


TIME_SERIES_RISK_QUERY = """
SELECT 
    DATE_TRUNC('month', as_of_date) AS month,
    industry,
    AVG(risk_probability) AS avg_monthly_risk,
    COUNT(DISTINCT company_id) AS company_count,
    SUM(CASE WHEN risk_probability > 0.5 THEN 1 ELSE 0 END) AS high_risk_count
FROM risk_scored_data
WHERE as_of_date >= CURRENT_DATE - INTERVAL '12 months'
GROUP BY DATE_TRUNC('month', as_of_date), industry
ORDER BY month DESC, industry;
"""


VAR_BY_COMPANY_QUERY = """
SELECT 
    company_id,
    industry,
    AVG(daily_return) AS mean_return,
    STDDEV(daily_return) AS std_return,
    PERCENTILE_CONT(0.05) WITHIN GROUP (ORDER BY daily_return) AS var_95pct,
    PERCENTILE_CONT(0.01) WITHIN GROUP (ORDER BY daily_return) AS var_99pct
FROM market_data
WHERE date >= CURRENT_DATE - INTERVAL '90 days'
GROUP BY company_id, industry
ORDER BY var_95pct ASC;
"""


PNL_ATTRIBUTION_QUERY = """
SELECT 
    company_id,
    industry,
    SUM(pnl) AS total_pnl,
    SUM(pnl_pct) AS total_pnl_pct,
    AVG(equity_price) AS avg_price,
    AVG(yield) AS avg_yield,
    AVG(credit_spread) AS avg_spread,
    MAX(date) AS last_update_date
FROM pnl_data
WHERE date >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY company_id, industry
ORDER BY total_pnl DESC;
"""


TOP_RISK_DRIVERS_QUERY = """
SELECT 
    feature_name,
    AVG(shap_value) AS avg_shap_importance,
    STDDEV(shap_value) AS std_shap_importance
FROM shap_values
WHERE as_of_date = (SELECT MAX(as_of_date) FROM shap_values)
GROUP BY feature_name
ORDER BY AVG(ABS(shap_value)) DESC
LIMIT 10;
"""


def execute_sql_query(engine, query: str, params: Optional[dict] = None) -> pd.DataFrame:
    """
    Execute a SQL query and return results as DataFrame.
    """
    with engine.connect() as conn:
        result = conn.execute(text(query), params or {})
        df = pd.DataFrame(result.fetchall(), columns=result.keys())
    return df


def create_risk_tables(engine):
    """
    Create tables and views for risk analytics (for demonstration).
    In production, these would be managed by DBA/migrations.
    """
    with engine.connect() as conn:
        conn.execute(text(RISK_SCORES_VIEW))
        conn.commit()


def get_high_risk_companies(engine) -> pd.DataFrame:
    """Get high-risk companies using SQL."""
    return execute_sql_query(engine, HIGH_RISK_COMPANIES_QUERY)


def get_industry_benchmarks(engine) -> pd.DataFrame:
    """Get industry benchmarks using SQL aggregation."""
    return execute_sql_query(engine, INDUSTRY_BENCHMARKS_QUERY)


def get_time_series_risk(engine) -> pd.DataFrame:
    """Get time-series risk trends using SQL window functions."""
    return execute_sql_query(engine, TIME_SERIES_RISK_QUERY)

