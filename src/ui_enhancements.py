"""
Enhanced UI components for better user experience and modern design.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Any


def set_page_config():
    """Set modern page configuration."""
    st.set_page_config(
        page_title="Enterprise Risk Intelligence Platform",
        page_icon="🏢",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': None,
            'Report a bug': None,
            'About': "Enterprise Risk Intelligence Platform v2.0"
        }
    )


def apply_custom_css():
    """Apply custom CSS for modern styling."""
    st.markdown("""
    <style>
    /* Main theme improvements */
    .main {
        padding-top: 2rem;
        background-color: #f8f9fa;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
    }
    
    /* Card styling */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(102, 126, 234, 0.3);
    }
    
    /* Data table styling */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Chart container */
    .chart-container {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    /* Alert styling */
    .alert-info {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* Header styling */
    .header-title {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    
    /* Sidebar text */
    .sidebar-text {
        color: white;
        font-weight: 500;
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Select box styling */
    .stSelectbox > div > div {
        background: white;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Slider styling */
    .stSlider > div > div > div {
        background: #667eea;
    }
    
    /* Risk level indicators */
    .risk-low { color: #10b981; font-weight: 600; }
    .risk-medium { color: #f59e0b; font-weight: 600; }
    .risk-high { color: #ef4444; font-weight: 600; }
    </style>
    """, unsafe_allow_html=True)


def create_metric_card(title: str, value: str, delta: str = None, icon: str = None, color: str = "#667eea"):
    """Create modern metric card with styling."""
    if icon:
        title = f"{icon} {title}"
    
    if delta:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: {color}; font-size: 1.2rem; margin-bottom: 0.5rem;">{title}</h3>
            <p style="font-size: 2rem; font-weight: 700; margin: 0;">{value}</p>
            <p style="color: {color}; font-size: 1rem; margin: 0;">{delta}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: {color}; font-size: 1.2rem; margin-bottom: 0.5rem;">{title}</h3>
            <p style="font-size: 2rem; font-weight: 700; margin: 0;">{value}</p>
        </div>
        """, unsafe_allow_html=True)


def create_info_box(message: str, type: str = "info"):
    """Create styled info box."""
    colors = {
        "info": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
        "success": "linear-gradient(135deg, #10b981 0%, #059669 100%)",
        "warning": "linear-gradient(135deg, #f59e0b 0%, #d97706 100%)",
        "error": "linear-gradient(135deg, #ef4444 0%, #dc2626 100%)"
    }
    
    icons = {
        "info": "ℹ️",
        "success": "✅",
        "warning": "⚠️",
        "error": "❌"
    }
    
    st.markdown(f"""
    <div class="alert-info" style="background: {colors.get(type, colors['info'])};">
        <strong>{icons.get(type, 'ℹ️')} {message}</strong>
    </div>
    """, unsafe_allow_html=True)


def create_enhanced_chart(fig, title: str = None, use_container_width: bool = True):
    """Create enhanced chart container with styling."""
    if title:
        fig.update_layout(
            title={
                'text': title,
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'color': '#1f2937'}
            }
        )
    
    # Apply modern theme
    fig.update_layout(
        template="plotly_white",
        font=dict(color="#1f2937"),
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=20, r=20, t=60, b=20),
        showlegend=True,
        legend=dict(
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="#e5e7eb",
            borderwidth=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=use_container_width)


def create_risk_gauge(value: float, title: str = "Risk Score"):
    """Create modern gauge chart for risk visualization."""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title},
        delta = {'reference': 0.5},
        gauge = {
            'axis': {'range': [None, 1]},
            'bar': {'color': "#667eea"},
            'steps': [
                {'range': [0, 0.3], 'color': "#10b981"},
                {'range': [0.3, 0.7], 'color': "#f59e0b"},
                {'range': [0.7, 1], 'color': "#ef4444"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 0.8
            }
        }
    ))
    
    fig.update_layout(
        template="plotly_white",
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig


def create_progress_bar(progress: float, title: str = "Progress"):
    """Create styled progress bar."""
    color = "#10b981" if progress >= 0.8 else "#f59e0b" if progress >= 0.5 else "#ef4444"
    
    st.markdown(f"""
    <div style="margin: 1rem 0;">
        <h4 style="margin-bottom: 0.5rem;">{title}</h4>
        <div style="background: #e5e7eb; border-radius: 10px; height: 30px; overflow: hidden;">
            <div style="background: {color}; width: {progress*100}%; height: 100%; display: flex; align-items: center; justify-content: center; color: white; font-weight: 600;">
                {progress*100:.1f}%
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def create_sidebar_navigation():
    """Create enhanced sidebar navigation."""
    with st.sidebar:
        # Logo/Title
        st.markdown("""
        <div style="text-align: center; padding: 2rem 0;">
            <h1 style="color: white; font-size: 1.5rem; margin: 0;">🏢 ERIP</h1>
            <p style="color: rgba(255,255,255,0.8); margin: 0.5rem 0;">Risk Intelligence Platform</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Navigation
        st.markdown('<p class="sidebar-text">📋 Navigation</p>', unsafe_allow_html=True)
        
        views = [
            "Portfolio Overview",
            "Peer Benchmarking", 
            "Explainability (SHAP)",
            "Scenario & Stress Testing",
            "Risk Recommendations",
            "Market Risk & VaR",
            "P&L Attribution",
            "SQL Queries & Export"
        ]
        
        selected_view = st.selectbox("Select View", views, key="navigation")
        
        st.markdown("---")
        
        # Quick Stats
        st.markdown('<p class="sidebar-text">📊 Quick Stats</p>', unsafe_allow_html=True)
        
        # This would be populated with actual stats
        st.metric("Companies Analyzed", "2,144")
        st.metric("Risk Models", "2")
        st.metric("Data Sources", "4")
        
        return selected_view


def create_filter_section(df: pd.DataFrame):
    """Create enhanced filter section."""
    st.markdown("### 🔍 Filters & Controls")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        industries = ["All"] + sorted(df['industry'].unique().astype(str))
        selected_industry = st.selectbox("Industry", industries)
    
    with col2:
        regions = ["All"] + sorted(df['region'].unique().astype(str))
        selected_region = st.selectbox("Region", regions)
    
    with col3:
        risk_threshold = st.slider("Risk Threshold", 0.0, 1.0, 0.5, 0.05)
    
    with col4:
        sort_by = st.selectbox("Sort By", ["Risk Score", "Company Name", "Industry"])
    
    return selected_industry, selected_region, risk_threshold, sort_by


def create_data_table_enhanced(df: pd.DataFrame, title: str = "Data Table"):
    """Create enhanced data table with styling."""
    st.markdown(f"### 📋 {title}")
    
    # Add risk level styling
    def format_risk(val):
        if val >= 0.7:
            return f'<span class="risk-high">{val:.3f}</span>'
        elif val >= 0.4:
            return f'<span class="risk-medium">{val:.3f}</span>'
        else:
            return f'<span class="risk-low">{val:.3f}</span>'
    
    # Format the dataframe
    styled_df = df.copy()
    if 'predicted_proba' in styled_df.columns:
        styled_df['predicted_proba'] = styled_df['predicted_proba'].apply(format_risk)
    
    st.write(styled_df.to_html(escape=False), unsafe_allow_html=True)


def create_comparison_chart(df1: pd.DataFrame, df2: pd.DataFrame, title: str):
    """Create side-by-side comparison chart."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Before', 'After'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Add traces for both datasets
    fig.add_trace(
        go.Histogram(x=df1['predicted_proba'], name='Before', marker_color='#ef4444'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Histogram(x=df2['predicted_proba'], name='After', marker_color='#10b981'),
        row=1, col=2
    )
    
    fig.update_layout(
        title_text=title,
        template="plotly_white",
        showlegend=True,
        height=400
    )
    
    return fig


def create_trend_chart(df: pd.DataFrame, date_col: str, value_col: str, title: str):
    """Create enhanced trend chart with confidence bands."""
    fig = go.Figure()
    
    # Main trend line
    fig.add_trace(go.Scatter(
        x=df[date_col],
        y=df[value_col],
        mode='lines+markers',
        name='Trend',
        line=dict(color='#667eea', width=3),
        marker=dict(size=6)
    ))
    
    # Add moving average
    if len(df) > 7:
        df['MA'] = df[value_col].rolling(window=7).mean()
        fig.add_trace(go.Scatter(
            x=df[date_col],
            y=df['MA'],
            mode='lines',
            name='7-Day MA',
            line=dict(color='#764ba2', width=2, dash='dash')
        ))
    
    fig.update_layout(
        title=title,
        template="plotly_white",
        xaxis_title="Date",
        yaxis_title="Value",
        hovermode='x unified',
        height=400
    )
    
    return fig


def create_heatmap_enhanced(corr_matrix: pd.DataFrame, title: str = "Correlation Heatmap"):
    """Create enhanced correlation heatmap."""
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.round(2).values,
        texttemplate="%{text}",
        textfont={"size": 10},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title=title,
        template="plotly_white",
        width=800,
        height=600
    )
    
    return fig
