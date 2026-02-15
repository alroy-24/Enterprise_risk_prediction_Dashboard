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
from improve_charts import (
    create_risk_distribution_chart, create_industry_risk_chart,
    create_financial_health_chart, create_risk_scatter_chart,
    create_correlation_heatmap, create_portfolio_summary_cards, create_trend_chart
)


st.set_page_config(
    page_title="Enterprise Risk Intelligence Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/RiskPrediction',
        'Report a bug': None,
        'About': "# Enterprise Risk Intelligence Platform\nAI-powered risk analytics for financial institutions."
    }
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
    }
    
    /* Metric cards enhancement */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 600;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e1e2e 0%, #262730 100%);
    }
    
    /* Card-like containers */
    .stAlert {
        border-radius: 10px;
        border-left: 4px solid #667eea;
    }
    
    /* Button styling */
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        border: none;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* Dataframe styling */
    .dataframe {
        border-radius: 8px;
    }
    
    /* Success/Warning/Error messages */
    .element-container .stSuccess {
        background-color: rgba(0, 255, 0, 0.1);
        border-left: 4px solid #00ff00;
    }
    
    .element-container .stWarning {
        background-color: rgba(255, 165, 0, 0.1);
        border-left: 4px solid #ffa500;
    }
    
    .element-container .stError {
        background-color: rgba(255, 0, 0, 0.1);
        border-left: 4px solid #ff0000;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        font-weight: 600;
    }
    
    /* Risk badge styling */
    .risk-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    
    .risk-high {
        background-color: #ff4444;
        color: white;
    }
    
    .risk-medium {
        background-color: #ffaa00;
        color: white;
    }
    
    .risk-low {
        background-color: #00cc66;
        color: white;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Improve spacing */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_dataset():
    # Try fixed expanded dataset first, fallback to expanded, then original
    with st.spinner("🔄 Loading dataset..."):
        try:
            df = pd.read_csv("data/expanded_real_world_financials_fixed.csv")
            st.success("✅ Loaded expanded real-world dataset (2,144 records)", icon="✅")
        except FileNotFoundError:
            try:
                df = pd.read_csv("data/expanded_real_world_financials.csv")
                st.info("📊 Loaded expanded real-world dataset (2,144 records)", icon="📊")
            except FileNotFoundError:
                try:
                    df = pd.read_csv("data/real_world_financials.csv")
                    st.info("📊 Loaded real-world dataset (15 records)", icon="📊")
                except FileNotFoundError:
                    # Fallback to original config-based loading
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
                    st.info("📊 Loaded sample dataset from config", icon="📊")
    
    # Load config separately
    try:
        cfg = config.load_model_config()
    except:
        # Default config if loading fails
        cfg = {
            "data": {"source": "data/sample_financials.csv", "target": "risk_flag"},
            "model": {"random_state": 42},
            "scenarios": {},
            "weights": {}
        }
    
    # Ensure target column exists
    if "risk_flag" not in df.columns:
        df["risk_flag"] = 0
        st.warning("⚠️ Risk flag column not found, setting all to 0")
    
    df = engineer_features(df)
    return df, cfg


@st.cache_resource
def train():
    with st.spinner("🤖 Training ML models..."):
        df, cfg = load_dataset()
        trained = train_models(df, cfg)
        st.success("✅ ML models trained successfully!", icon="🎯")
    return trained


def get_predictions(trained, df_feat: pd.DataFrame):
    pipeline = get_inference_pipeline(trained)
    X = df_feat[trained.feature_cols]
    proba = pipeline.predict_proba(X)[:, 1]
    df_feat = df_feat.copy()
    df_feat["predicted_proba"] = proba
    return df_feat, pipeline


def layout_header(df_scored: pd.DataFrame):
    # Enhanced header with gradient styling
    st.markdown('<h1 class="main-header">🏢 Enterprise Risk Intelligence Platform</h1>', unsafe_allow_html=True)
    
    # Subtitle with badges
    st.markdown("""
        <div style="margin-bottom: 2rem;">
            <span style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                         color: white; padding: 6px 12px; border-radius: 20px; 
                         font-size: 0.85rem; font-weight: 600; margin-right: 8px;">
                🤖 Explainable AI
            </span>
            <span style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                         color: white; padding: 6px 12px; border-radius: 20px; 
                         font-size: 0.85rem; font-weight: 600; margin-right: 8px;">
                📊 Market Risk Analytics
            </span>
            <span style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                         color: white; padding: 6px 12px; border-radius: 20px; 
                         font-size: 0.85rem; font-weight: 600;">
                🏦 Enterprise-Grade
            </span>
        </div>
    """, unsafe_allow_html=True)
    
    # Dynamic metrics based on actual data
    create_portfolio_summary_cards(df_scored)


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
    
    # Filter summary
    st.markdown(f"""
        <div style="padding: 0.75rem; background: rgba(102, 126, 234, 0.1); 
                    border-radius: 8px; margin-bottom: 1rem; border-left: 4px solid #667eea;">
            📋 <strong>Filtered Results:</strong> {len(df_filtered)} companies | 
            Industry: <strong>{selected_industry}</strong> | 
            Region: <strong>{selected_region}</strong> | 
            Risk Threshold: <strong>{risk_threshold:.0%}</strong>
        </div>
    """, unsafe_allow_html=True)
    
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
        height=400,
        column_config={
            "predicted_proba": st.column_config.ProgressColumn(
                "Risk Score",
                format="%.1f%%",
                min_value=0,
                max_value=1,
            ),
            "risk_rating": st.column_config.TextColumn(
                "Rating",
                help="Credit-style risk rating"
            )
        }
    )
    
    # Improved Visualizations
    st.markdown("---")
    st.subheader("📈 Risk Analytics Dashboard")
    
    # Row 1: Distribution and Industry Comparison
    col1, col2 = st.columns(2)
    
    with col1:
        # Risk distribution
        fig_dist = create_risk_distribution_chart(df_filtered, "Risk Probability Distribution")
        st.plotly_chart(fig_dist, use_container_width=True)
    
    with col2:
        # Industry risk comparison
        fig_industry = create_industry_risk_chart(df_filtered)
        st.plotly_chart(fig_industry, use_container_width=True)
    
    # Row 2: Risk vs Leverage and Financial Health
    col1, col2 = st.columns(2)
    
    with col1:
        # Risk vs Leverage scatter plot
        fig_scatter = create_risk_scatter_chart(df_filtered)
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with col2:
        # Financial health radar (top 5 industries)
        fig_radar = create_financial_health_chart(df_filtered)
        st.plotly_chart(fig_radar, use_container_width=True)
    
    # Row 3: Correlation Heatmap
    st.subheader("🔗 Risk Factor Correlations")
    fig_corr = create_correlation_heatmap(df_filtered)
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # Trend analysis if date data available
    if 'as_of_date' in df_filtered.columns:
        st.subheader("📊 Risk Trends Over Time")
        fig_trend = create_trend_chart(df_filtered)
        if fig_trend:
            st.plotly_chart(fig_trend, use_container_width=True)


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
    """Generate market data (cached for performance) using real companies."""
    # Load the dataset to get real companies
    try:
        df = pd.read_csv("data/expanded_real_world_financials_fixed.csv")
    except:
        df = pd.read_csv("data/expanded_real_world_financials.csv")
    
    # Generate market data with real companies
    dates = pd.date_range(start="2024-01-01", end="2024-12-31", freq="D")
    companies = df['company_id'].unique()
    
    data = []
    np.random.seed(42)
    
    for company in companies:
        # Get company-specific data for realistic parameters
        company_data = df[df['company_id'] == company].iloc[0]
        
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


def market_risk_section(df_scored: pd.DataFrame):
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


def pnl_section(df_scored: pd.DataFrame):
    """
    P&L Attribution Dashboard - Detailed profit and loss analysis.
    """
    st.subheader("💰 P&L Attribution Analysis")
    st.caption("Comprehensive profit and loss attribution by company and factors")
    
    # Generate market data
    try:
        market_df = generate_market_data_cached()
        
        # Calculate returns
        try:
            market_df_returns = calculate_returns(market_df)
        except Exception as e:
            st.error(f"Error calculating returns: {e}")
            return
        
        # PnL Attribution
        st.subheader("📊 P&L by Company")
        
        if "equity_price" in market_df_returns.columns:
            pnl_results = calculate_pnl(market_df_returns)
            
            if len(pnl_results) > 0:
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total P&L", f"${pnl_results['pnl'].sum():,.0f}")
                with col2:
                    st.metric("Average P&L", f"${pnl_results['pnl'].mean():,.0f}")
                with col3:
                    winners = (pnl_results['pnl'] > 0).sum()
                    st.metric("Winning Positions", f"{winners}/{len(pnl_results)}")
                with col4:
                    win_rate = (pnl_results['pnl'] > 0).mean() * 100
                    st.metric("Win Rate", f"{win_rate:.1f}%")
                
                # Detailed table
                st.dataframe(pnl_results.sort_values("pnl", ascending=False), use_container_width=True)
                
                # Visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.bar(
                        pnl_results.head(15),
                        x="company_id",
                        y="pnl",
                        title="Top 15 P&L by Company",
                        labels={"pnl": "P&L ($)", "company_id": "Company ID"},
                        color="pnl",
                        color_continuous_scale=["red", "yellow", "green"]
                    )
                    fig.update_xaxes(tickangle=45)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.bar(
                        pnl_results.head(15),
                        x="company_id",
                        y="pnl_pct",
                        title="Top 15 P&L % by Company",
                        labels={"pnl_pct": "P&L (%)", "company_id": "Company ID"},
                        color="pnl_pct",
                        color_continuous_scale=["red", "yellow", "green"]
                    )
                    fig.update_xaxes(tickangle=45)
                    st.plotly_chart(fig, use_container_width=True)
                
                # P&L Distribution
                st.subheader("📈 P&L Distribution")
                fig = px.histogram(
                    pnl_results,
                    x="pnl",
                    nbins=30,
                    title="P&L Distribution Across Portfolio",
                    labels={"pnl": "P&L ($)"}
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Factor Attribution (if available)
                st.subheader("🔍 P&L Factor Attribution")
                try:
                    if "daily_return" in market_df_returns.columns:
                        factor_attr = attribute_pnl_by_factor(market_df_returns)
                        if len(factor_attr) > 0:
                            st.dataframe(factor_attr, use_container_width=True)
                        else:
                            st.info("Factor attribution analysis requires additional market factors")
                except Exception as e:
                    st.info(f"Factor attribution not available: {e}")
            else:
                st.warning("No P&L data available")
        else:
            st.error("Market data missing required columns for P&L calculation")
    
    except Exception as e:
        st.error(f"P&L analysis failed: {e}")
        st.info("This feature requires market data. In production, this would connect to real market data feeds.")


def sql_section(df_scored: pd.DataFrame):
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
        st.info("💡 **In production**, these queries would execute against your PostgreSQL database using `sql_queries` module.")
        
        # Show how to execute
        with st.expander("How to Execute (Python Code)"):
            st.code(f"""
from sql_queries import get_high_risk_companies, get_industry_benchmarks
from sqlalchemy import create_engine

engine = create_engine("postgresql://user:pass@host:5432/db")
df = get_{query_type.lower().replace(' ', '_')}(engine)
            """, language="python")
    
    # Export options
    st.subheader("📤 Export Options")
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


def main():
    df, cfg = load_dataset()
    trained = train()
    df_feat, pipeline = get_predictions(trained, df)
    
    # Load weights properly from config
    try:
        weights = config.load_weights()
        # Convert weights format if needed
        if "financial_weight" in weights:
            cfg["weights"] = {
                "financial": weights["financial_weight"],
                "operational": weights["operational_weight"], 
                "compliance": weights["compliance_weight"]
            }
        else:
            cfg["weights"] = weights
    except Exception as e:
        st.warning(f"Could not load weights config: {e}")
        cfg["weights"] = {
            "financial": 0.45,
            "operational": 0.35,
            "compliance": 0.20
        }
    
    df_scored = aggregate_scores(df_feat, cfg["weights"])
    
    # Dynamic header with actual data
    layout_header(df_scored)
    
    # Enhanced Sidebar Navigation
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎯 Navigation")
    
    # Add sidebar info
    with st.sidebar:
        st.markdown("""
            <div style="padding: 1rem; background: rgba(102, 126, 234, 0.1); 
                        border-radius: 8px; margin-bottom: 1rem;">
                <p style="margin: 0; font-size: 0.85rem; color: #888;">
                    <strong>Platform Status:</strong><br>
                    ✅ ML Models Trained<br>
                    ✅ {companies} Companies Analyzed<br>
                    ✅ Real-time Analytics
                </p>
            </div>
        """.format(companies=len(df_scored)), unsafe_allow_html=True)
    
    # Navigation with icons
    view_options = {
        "📊 Portfolio Overview": "Portfolio Overview",
        "🏆 Peer Benchmarking": "Peer Benchmarking",
        "🔍 Explainability (SHAP)": "Explainability (SHAP)",
        "⚡ Scenario & Stress Testing": "Scenario & Stress Testing",
        "💡 Risk Recommendations": "Risk Recommendations",
        "📈 Market Risk & VaR": "Market Risk & VaR",
        "💰 P&L Attribution": "P&L Attribution",
        "🗄️ SQL Queries & Export": "SQL Queries & Export"
    }
    
    view_display = st.sidebar.radio(
        "Select Analysis View:",
        list(view_options.keys()),
        label_visibility="collapsed"
    )
    view = view_options[view_display]
    
    # Add footer to sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
        <div style="text-align: center; color: #888; font-size: 0.75rem; padding: 1rem;">
            <strong>ERIP v2.0</strong><br>
            Enterprise Risk Intelligence<br>
            Powered by ML & Advanced Analytics
        </div>
    """, unsafe_allow_html=True)
    
    if view == "Portfolio Overview":
        portfolio_view(df_scored)
    elif view == "Peer Benchmarking":
        peer_benchmarking_section(df_scored)
    elif view == "Explainability (SHAP)":
        shap_section(df_feat, pipeline, trained)
    elif view == "Scenario & Stress Testing":
        scenario_section(df_feat, pipeline, trained)
    elif view == "Risk Recommendations":
        recommendations_section(df_scored)
    elif view == "Market Risk & VaR":
        market_risk_section(df_scored)
    elif view == "P&L Attribution":
        pnl_section(df_scored)
    elif view == "SQL Queries & Export":
        sql_section(df_scored)


if __name__ == "__main__":
    main()
