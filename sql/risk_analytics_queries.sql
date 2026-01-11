-- SQL Queries for Enterprise Risk Intelligence Platform
-- Industry-standard SQL for risk analytics
-- These queries demonstrate SQL skills: aggregations, window functions, CTEs, subqueries

-- ============================================
-- 1. Risk Scores View (Materialized View)
-- ============================================
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


-- ============================================
-- 2. High-Risk Companies Query
-- ============================================
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


-- ============================================
-- 3. Industry Benchmarks (Aggregation)
-- ============================================
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


-- ============================================
-- 4. Time-Series Risk Trends (Window Functions)
-- ============================================
SELECT 
    DATE_TRUNC('month', as_of_date) AS month,
    industry,
    AVG(risk_probability) AS avg_monthly_risk,
    COUNT(DISTINCT company_id) AS company_count,
    SUM(CASE WHEN risk_probability > 0.5 THEN 1 ELSE 0 END) AS high_risk_count,
    LAG(AVG(risk_probability)) OVER (PARTITION BY industry ORDER BY DATE_TRUNC('month', as_of_date)) AS prev_month_risk
FROM risk_scored_data
WHERE as_of_date >= CURRENT_DATE - INTERVAL '12 months'
GROUP BY DATE_TRUNC('month', as_of_date), industry
ORDER BY month DESC, industry;


-- ============================================
-- 5. VaR by Company (Statistical Functions)
-- ============================================
SELECT 
    company_id,
    industry,
    AVG(daily_return) AS mean_return,
    STDDEV(daily_return) AS std_return,
    PERCENTILE_CONT(0.05) WITHIN GROUP (ORDER BY daily_return) AS var_95pct,
    PERCENTILE_CONT(0.01) WITHIN GROUP (ORDER BY daily_return) AS var_99pct,
    MIN(daily_return) AS worst_day_return,
    MAX(daily_return) AS best_day_return
FROM market_data
WHERE date >= CURRENT_DATE - INTERVAL '90 days'
GROUP BY company_id, industry
ORDER BY var_95pct ASC;


-- ============================================
-- 6. PnL Attribution (CTE & Joins)
-- ============================================
WITH latest_prices AS (
    SELECT 
        company_id,
        equity_price AS current_price,
        date AS price_date
    FROM market_data
    WHERE date = (SELECT MAX(date) FROM market_data)
),
previous_prices AS (
    SELECT 
        company_id,
        equity_price AS previous_price,
        date AS price_date
    FROM market_data
    WHERE date = (
        SELECT MAX(date) FROM market_data 
        WHERE date < (SELECT MAX(date) FROM market_data)
    )
)
SELECT 
    lp.company_id,
    rsv.industry,
    lp.current_price - pp.previous_price AS price_change,
    (lp.current_price / pp.previous_price - 1) * 100 AS pnl_pct,
    rsv.risk_probability,
    rsv.leverage_ratio
FROM latest_prices lp
JOIN previous_prices pp ON lp.company_id = pp.company_id
LEFT JOIN risk_scores_view rsv ON lp.company_id = rsv.company_id
ORDER BY pnl_pct DESC;


-- ============================================
-- 7. Top Risk Drivers (Window Functions)
-- ============================================
SELECT 
    feature_name,
    AVG(shap_value) AS avg_shap_importance,
    STDDEV(shap_value) AS std_shap_importance,
    ROW_NUMBER() OVER (ORDER BY AVG(ABS(shap_value)) DESC) AS importance_rank
FROM shap_values
WHERE as_of_date = (SELECT MAX(as_of_date) FROM shap_values)
GROUP BY feature_name
ORDER BY AVG(ABS(shap_value)) DESC
LIMIT 10;


-- ============================================
-- 8. Portfolio Risk Summary (Multiple Aggregations)
-- ============================================
SELECT 
    COUNT(DISTINCT company_id) AS total_companies,
    COUNT(DISTINCT industry) AS total_industries,
    AVG(risk_probability) AS portfolio_avg_risk,
    SUM(CASE WHEN risk_rating IN ('AAA', 'AA', 'A') THEN 1 ELSE 0 END) AS investment_grade_count,
    SUM(CASE WHEN risk_rating IN ('BBB', 'BB', 'B') THEN 1 ELSE 0 END) AS speculative_grade_count,
    SUM(CASE WHEN risk_rating IN ('CCC', 'CC', 'C', 'D') THEN 1 ELSE 0 END) AS high_risk_count,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY risk_probability) AS portfolio_95th_percentile_risk
FROM risk_scores_view;

