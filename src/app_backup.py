import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from pathlib import Path
from io import BytesIO

import config
from data_ingest import DataSource, load_data
from features import engineer_features
from models import train_models, get_inference_pipeline
from explainability import compute_shap_values, summarize_top_features
from scenarios import run_scenarios
from aggregation import aggregate_scores
from recommendations import generate_recommendations
from risk_rating import add_risk_ratings, get_rating_color
from benchmarking import compute_peer_benchmarks, add_peer_comparison, get_peer_summary
from market_data import generate_market_data, calculate_returns, merge_market_with_financials
from var_calculations import (
    parametric_var, historical_var, monte_carlo_var,
    calculate_portfolio_var, calculate_individual_var
)
from pnl_attribution import calculate_pnl, attribute_pnl_by_factor, calculate_portfolio_pnl
from sql_queries import (
    get_high_risk_companies, get_industry_benchmarks, get_time_series_risk,
    HIGH_RISK_COMPANIES_QUERY, INDUSTRY_BENCHMARKS_QUERY, TIME_SERIES_RISK_QUERY
)


st.set_page_config(page_title="Enterprise Risk Intelligence Platform", layout="wide")


@st.cache_data
def load_dataset():
    cfg = config.load_model_config()
    source = cfg["data"]["source"]
    if source.startswith("data/"):
        path = Path(source)
        ds = DataSource(path=path)
    elif source == "postgres":
        ds = DataSource(
            postgres_uri=cfg["data"]["postgres"]["uri"],
            table=cfg["data"]["postgres"]["table"],
        )
    else:
        raise ValueError("Unsupported data source.")
    df = load_data(ds)
    return df, cfg


@st.cache_resource
def train():
    df, cfg = load_dataset()
    trained = train_models(df, cfg)
    return trained


def get_predictions(trained, df_feat: pd.DataFrame):
    pipeline = get_inference_pipeline(trained)
    X = df_feat[trained.feature_cols]
    proba = pipeline.predict_proba(X)[:, 1]
    df_feat = df_feat.copy()
    df_feat["predicted_proba"] = proba
    return df_feat, pipeline


def layout_header():
    st.title("🏢 Enterprise Risk Intelligence Platform (ERIP)")
    st.caption("**Consulting-grade risk analytics** | Explainable AI | Regulatory stress testing | Peer benchmarking")
    
    # Key metrics summary
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Portfolio Size", "20 companies")
    with col2:
        st.metric("Industries", "5 sectors")
    with col3:
        st.metric("Regions", "3 regions")
    with col4:
        st.metric("Risk Models", "Logistic + XGBoost")


def portfolio_view(df_scored: pd.DataFrame):
    st.subheader("📊 Portfolio Risk Scorecard")
    
    # Add risk ratings
    df_scored = add_risk_ratings(df_scored)
    
    # Filter options
    col1, col2, col3 = st.columns(3)
    with col1:
        # Handle potential NaN/mixed types in industry
        industries_unique = df_scored["industry"].dropna().astype(str).unique().tolist()
        industries = ["All"] + sorted([i for i in industries_unique if i != "nan"])
        selected_industry = st.selectbox("Filter by Industry", industries)
    with col2:
        # Handle potential NaN/mixed types in region
        regions_unique = df_scored["region"].dropna().astype(str).unique().tolist()
        regions = ["All"] + sorted([r for r in regions_unique if r != "nan"])
        selected_region = st.selectbox("Filter by Region", regions)
    with col3:
        risk_threshold = st.slider("Risk Threshold", 0.0, 1.0, 0.5, 0.05)
    
    # Apply filters
    df_filtered = df_scored.copy()
    if selected_industry != "All":
        df_filtered = df_filtered[df_filtered["industry"] == selected_industry]
    if selected_region != "All":
        df_filtered = df_filtered[df_filtered["region"] == selected_region]
    df_filtered = df_filtered[df_filtered["predicted_proba"] >= risk_threshold]
    
    # Display table with key metrics
    display_cols = [
        "company_id", "industry", "region", "risk_rating", 
        "predicted_proba", "leverage_ratio", "current_ratio", 
        "interest_coverage", "altman_z_score", "aggregated_risk"
    ]
    display_cols = [c for c in display_cols if c in df_filtered.columns]
    
    st.dataframe(
        df_filtered[display_cols].sort_values("predicted_proba", ascending=False),
        use_container_width=True,
        height=400
    )
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Risk probability by company (with ratings)
        fig = px.bar(
            df_filtered.sort_values("predicted_proba", ascending=False).head(15),
            x="company_id",
            y="predicted_proba",
            color="risk_rating",
            title="Risk Probability by Company (Top 15)",
            labels={"predicted_proba": "Risk Probability", "company_id": "Company ID"},
            color_discrete_map={r: get_rating_color(r) for r in df_filtered["risk_rating"].unique()}
        )
        fig.update_layout(showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Risk distribution by industry
        fig = px.box(
            df_filtered,
            x="industry",
            y="predicted_proba",
            title="Risk Distribution by Industry",
            labels={"predicted_proba": "Risk Probability", "industry": "Industry"}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Risk heatmap
    st.subheader("🔥 Risk Heatmap: Key Financial Ratios")
    heatmap_cols = ["leverage_ratio", "current_ratio", "interest_coverage", "ebitda_margin", "altman_z_score"]
    heatmap_cols = [c for c in heatmap_cols if c in df_filtered.columns]
    
    if heatmap_cols:
        heatmap_data = df_filtered.set_index("company_id")[heatmap_cols]
        # Normalize for visualization
        heatmap_data_norm = (heatmap_data - heatmap_data.min()) / (heatmap_data.max() - heatmap_data.min() + 1e-6)
        
        fig = px.imshow(
            heatmap_data_norm.T,
            labels=dict(x="Company", y="Metric", color="Normalized Value"),
            x=heatmap_data.index.tolist(),
            y=heatmap_cols,
            aspect="auto",
            color_continuous_scale="RdYlGn_r",
            title="Financial Ratios Heatmap (Red = Higher Risk)"
        )
        st.plotly_chart(fig, use_container_width=True)


def peer_benchmarking_section(df_scored: pd.DataFrame):
    st.subheader("📈 Peer Benchmarking & Industry Analysis")
    
    df_scored = add_peer_comparison(df_scored, group_by="industry")
    
    # Select company for detailed peer comparison
    selected_company = st.selectbox("Select Company for Peer Analysis", df_scored["company_id"])
    
    if selected_company:
        peer_summary = get_peer_summary(df_scored, selected_company, group_by="industry")
        
        if peer_summary and "message" not in peer_summary:
            st.info(f"**{selected_company}** | Industry: **{peer_summary.get('industry', 'N/A')}** | Peer Count: **{peer_summary.get('peer_count', 0)}**")
            
            # Peer comparison metrics
            metrics_to_show = ["leverage_ratio", "current_ratio", "ebitda_margin", "predicted_proba"]
            metrics_to_show = [m for m in metrics_to_show if f"{m}_company" in peer_summary]
            
            if metrics_to_show:
                comparison_data = []
                for metric in metrics_to_show:
                    comparison_data.append({
                        "Metric": metric.replace("_", " ").title(),
                        "Company": peer_summary.get(f"{metric}_company", 0),
                        "Peer Average": peer_summary.get(f"{metric}_peer_avg", 0),
                        "Peer Median": peer_summary.get(f"{metric}_peer_median", 0),
                        "vs Average": peer_summary.get(f"{metric}_vs_avg", 0),
                        "Percentile": peer_summary.get(f"{metric}_percentile", "N/A")
                    })
                
                comp_df = pd.DataFrame(comparison_data)
                st.dataframe(comp_df, use_container_width=True)
                
                # Visualization
                fig = go.Figure()
                for metric in metrics_to_show:
                    fig.add_trace(go.Bar(
                        name=metric.replace("_", " ").title(),
                        x=["Company", "Peer Avg", "Peer Median"],
                        y=[
                            peer_summary.get(f"{metric}_company", 0),
                            peer_summary.get(f"{metric}_peer_avg", 0),
                            peer_summary.get(f"{metric}_peer_median", 0)
                        ]
                    ))
                fig.update_layout(
                    barmode="group",
                    title=f"Peer Comparison: {selected_company}",
                    xaxis_title="",
                    yaxis_title="Value"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Industry benchmarks table
        st.subheader("Industry Benchmarks")
        benchmarks = compute_peer_benchmarks(df_scored, group_by="industry")
        st.dataframe(benchmarks, use_container_width=True)


def shap_section(df_feat: pd.DataFrame, pipeline, trained):
    st.subheader("🔍 Explainability (SHAP Analysis)")
    try:
        shap_vals, shap_exp = compute_shap_values(pipeline, df_feat[trained.feature_cols])
        feature_names = pipeline.named_steps["prep"].get_feature_names_out()
        top_df = summarize_top_features(shap_exp, feature_names.tolist(), top_n=10)
        
        st.dataframe(top_df, use_container_width=True)
        
        # Visualization
        fig = px.bar(
            top_df.head(10),
            x="feature",
            y="mean_abs_shap",
            title="Top 10 Most Important Risk Features (SHAP)",
            labels={"mean_abs_shap": "SHAP Importance", "feature": "Feature"}
        )
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as exc:
        st.warning(f"SHAP could not run on this sample: {exc}")


def scenario_section(df_feat: pd.DataFrame, pipeline, trained):
    st.subheader("⚡ Scenario & Stress Testing")
    st.caption("Regulatory stress scenarios (Basel III, IFRS 9) and custom business scenarios")
    
    scenarios = config.load_scenarios()

    def infer_fn(shocked_df: pd.DataFrame):
        shocked_feat = engineer_features(shocked_df)
        proba = pipeline.predict_proba(shocked_feat[trained.feature_cols])[:, 1]
        return proba.mean()

    results = run_scenarios(df_feat, scenarios, infer_fn)
    res_df = pd.DataFrame(
        {
            "scenario": list(results.keys()),
            "description": [v["description"] for v in results.values()],
            "mean_probability": [v["probability"] for v in results.values()],
        }
    )
    
    # Add baseline comparison
    baseline_prob = res_df[res_df["scenario"] == "baseline"]["mean_probability"].values[0] if "baseline" in res_df["scenario"].values else 0
    res_df["vs_baseline"] = res_df["mean_probability"] - baseline_prob
    res_df["vs_baseline_pct"] = ((res_df["mean_probability"] - baseline_prob) / (baseline_prob + 1e-6)) * 100
    
    st.dataframe(res_df, use_container_width=True)
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(
            res_df,
            x="scenario",
            y="mean_probability",
            title="Average Risk Probability Under Scenarios",
            labels={"mean_probability": "Risk Probability", "scenario": "Scenario"}
        )
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(
            res_df[res_df["scenario"] != "baseline"],
            x="scenario",
            y="vs_baseline_pct",
            title="Impact vs Baseline (%)",
            labels={"vs_baseline_pct": "Change vs Baseline (%)", "scenario": "Scenario"},
            color="vs_baseline_pct",
            color_continuous_scale="Reds"
        )
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)


def recommendations_section(df_scored: pd.DataFrame):
    st.subheader("💡 Risk Recommendations & Remediation")
    selected = st.selectbox("Select company for recommendations", df_scored["company_id"])
    
    if selected:
        row = df_scored[df_scored["company_id"] == selected].iloc[0]
        recs = generate_recommendations(row)
        
        # Display key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Risk Rating", row.get("risk_rating", "N/A"))
        with col2:
            st.metric("Risk Probability", f"{row.get('predicted_proba', 0):.2%}")
        with col3:
            st.metric("Leverage Ratio", f"{row.get('leverage_ratio', 0):.2f}")
        with col4:
            st.metric("Current Ratio", f"{row.get('current_ratio', 0):.2f}")
        
        st.write("**Recommended Actions:**")
        for i, rec in enumerate(recs, 1):
            st.write(f"{i}. {rec}")


def export_section(df_scored: pd.DataFrame):
    st.subheader("📥 Export Reports")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Excel export
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df_scored.to_excel(writer, sheet_name='Risk Scores', index=False)
            # Add summary sheet
            summary = pd.DataFrame({
                "Metric": ["Total Companies", "High Risk Count", "Avg Risk Probability", "Industries"],
                "Value": [
                    len(df_scored),
                    len(df_scored[df_scored["predicted_proba"] > 0.5]),
                    df_scored["predicted_proba"].mean(),
                    df_scored["industry"].nunique()
                ]
            })
            summary.to_excel(writer, sheet_name='Summary', index=False)
        
        st.download_button(
            label="📊 Download Excel Report",
            data=output.getvalue(),
            file_name="erip_risk_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
    with col2:
        # CSV export
        csv = df_scored.to_csv(index=False)
        st.download_button(
            label="📄 Download CSV",
            data=csv,
            file_name="erip_risk_scores.csv",
            mime="text/csv"
        )


@st.cache_data
def generate_market_data_cached():
    """Generate market data (cached for performance)."""
    return generate_market_data(n_companies=20)


def market_risk_dashboard(df_scored: pd.DataFrame):
    """
    Market Risk Dashboard - Industry-standard analysis.
    Includes VaR, PnL attribution, market data visualization.
    """
    st.subheader("📈 Market Risk & Time-Series Analysis")
    st.caption("**Industry-standard** market risk metrics: VaR, PnL attribution, equity/fixed income analysis")
    
    # Generate market data
    try:
        market_df = generate_market_data_cached()
        
        # Ensure required columns exist
        if "equity_price" not in market_df.columns:
            st.error("Market data missing required columns. Please regenerate market data.")
            return
        
        # Calculate returns with error handling
        try:
            market_df_returns = calculate_returns(market_df)
        except Exception as e:
            st.error(f"Error calculating returns: {e}")
            st.info("Using market data without returns calculations.")
            market_df_returns = market_df.copy()
        
        # Merge with financial data (optional, not required for VaR/PnL)
        try:
            df_with_market = merge_market_with_financials(market_df_returns, df_scored)
        except Exception as e:
            st.warning(f"Could not merge market data with financials: {e}")
            df_with_market = df_scored.copy()
        
        # VaR Section
        st.subheader("💼 Value at Risk (VaR) Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            var_method = st.selectbox("VaR Method", ["parametric", "historical", "monte_carlo"])
            var_confidence = st.slider("Confidence Level", 0.90, 0.99, 0.95, 0.01)
        
        with col2:
            var_holding_period = st.slider("Holding Period (days)", 1, 30, 1)
        
        # Calculate individual VaR
        if "daily_return" in market_df_returns.columns:
            returns_pivot = market_df_returns.pivot_table(
                index="date", columns="company_id", values="daily_return"
            ).dropna()
            
            if len(returns_pivot) > 10:
                individual_var = calculate_individual_var(returns_pivot, method=var_method, confidence_level=var_confidence)
                
                if len(individual_var) > 0:
                    st.dataframe(individual_var.sort_values("var"), use_container_width=True)
                    
                    # VaR Visualization
                    fig = px.bar(
                        individual_var.head(15),
                        x="company_id",
                        y="var",
                        title=f"VaR by Company ({var_method.title()}, {var_confidence:.0%} confidence)",
                        labels={"var": "VaR", "company_id": "Company ID"}
                    )
                    fig.update_xaxes(tickangle=45)
                    st.plotly_chart(fig, use_container_width=True)
        
        # PnL Attribution Section
        st.subheader("💰 PnL Attribution")
        
        if "equity_price" in market_df_returns.columns:
            pnl_results = calculate_pnl(market_df_returns)
            
            if len(pnl_results) > 0:
                st.dataframe(pnl_results.sort_values("pnl", ascending=False), use_container_width=True)
                
                # PnL Visualization
                col1, col2 = st.columns(2)
                with col1:
                    fig = px.bar(
                        pnl_results.head(15),
                        x="company_id",
                        y="pnl",
                        title="PnL by Company",
                        labels={"pnl": "PnL ($)", "company_id": "Company ID"}
                    )
                    fig.update_xaxes(tickangle=45)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.bar(
                        pnl_results.head(15),
                        x="company_id",
                        y="pnl_pct",
                        title="PnL % by Company",
                        labels={"pnl_pct": "PnL (%)", "company_id": "Company ID"}
                    )
                    fig.update_xaxes(tickangle=45)
                    st.plotly_chart(fig, use_container_width=True)
        
        # Market Data Visualization
        st.subheader("📊 Market Data: Equity Prices & Yields")
        
        selected_company = st.selectbox("Select Company for Market Data", market_df_returns["company_id"].unique())
        
        if selected_company:
            company_market = market_df_returns[market_df_returns["company_id"] == selected_company].sort_values("date")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.line(
                    company_market,
                    x="date",
                    y="equity_price",
                    title=f"Equity Price Trend: {selected_company}",
                    labels={"equity_price": "Price ($)", "date": "Date"}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.line(
                    company_market,
                    x="date",
                    y=["yield", "credit_spread"],
                    title=f"Yield & Credit Spread: {selected_company}",
                    labels={"value": "Rate", "date": "Date"}
                )
                st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.warning(f"Market data generation failed: {e}")
        st.info("This feature requires market data. In production, this would connect to real market data feeds.")


def sql_queries_section():
    """SQL Queries Section - Show example SQL queries for risk analytics."""
    st.subheader("🗄️ SQL Queries for Risk Analytics")
    st.caption("**Industry-standard SQL** - Demonstrates SQL skills: aggregations, window functions, CTEs, subqueries")
    
    query_type = st.selectbox(
        "Select Query Type",
        ["High-Risk Companies", "Industry Benchmarks", "Time-Series Risk Trends", "VaR by Company", "PnL Attribution"]
    )
    
    queries = {
        "High-Risk Companies": HIGH_RISK_COMPANIES_QUERY,
        "Industry Benchmarks": INDUSTRY_BENCHMARKS_QUERY,
        "Time-Series Risk Trends": TIME_SERIES_RISK_QUERY,
    }
    
    if query_type in queries:
        st.code(queries[query_type], language="sql")
        st.info("💡 **In production**, these queries would execute against your PostgreSQL database using the `sql_queries` module.")
        
        # Show how to execute
        with st.expander("How to Execute (Python Code)"):
            st.code(f"""
from sql_queries import get_high_risk_companies, get_industry_benchmarks
from sqlalchemy import create_engine

engine = create_engine("postgresql://user:pass@host:5432/db")
df = get_{query_type.lower().replace(' ', '_')}(engine)
            """, language="python")


if __name__ == "__main__":
    layout_header()
    
    df_raw, cfg = load_dataset()
    trained = train()
    df_feat = engineer_features(df_raw)
    df_scored, pipeline = get_predictions(trained, df_feat)
    df_scored = aggregate_scores(df_scored, config.load_weights())
    df_scored = add_risk_ratings(df_scored)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select View",
        ["Portfolio Overview", "Market Risk", "SQL Queries", "Peer Benchmarking", "Explainability", "Stress Testing", "Recommendations", "Export"]
    )
    
    if page == "Portfolio Overview":
        portfolio_view(df_scored)
    elif page == "Market Risk":
        market_risk_dashboard(df_scored)
    elif page == "SQL Queries":
        sql_queries_section()
    elif page == "Peer Benchmarking":
        peer_benchmarking_section(df_scored)
    elif page == "Explainability":
        shap_section(df_feat, pipeline, trained)
    elif page == "Stress Testing":
        scenario_section(df_raw, pipeline, trained)
    elif page == "Recommendations":
        recommendations_section(df_scored)
    elif page == "Export":
        export_section(df_scored)
