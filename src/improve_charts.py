"""
Improved chart functions for better clarity and understanding.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np


def create_risk_distribution_chart(df, title="Risk Distribution"):
    """Create a clear risk distribution chart."""
    
    fig = go.Figure()
    
    # Add histogram
    fig.add_trace(go.Histogram(
        x=df['predicted_proba'],
        nbinsx=20,
        name='Risk Distribution',
        marker_color='rgba(55, 83, 109, 0.7)',
        opacity=0.8
    ))
    
    # Add mean line
    mean_risk = df['predicted_proba'].mean()
    fig.add_vline(
        x=mean_risk,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Mean: {mean_risk:.3f}",
        annotation_position="top right"
    )
    
    fig.update_layout(
        title=title,
        xaxis_title="Risk Probability",
        yaxis_title="Number of Companies",
        showlegend=True,
        template="plotly_white",
        height=400
    )
    
    return fig


def create_industry_risk_chart(df):
    """Create a clear industry risk comparison chart."""
    
    # Calculate industry averages
    industry_risk = df.groupby('industry')['predicted_proba'].agg(['mean', 'count', 'std']).reset_index()
    industry_risk = industry_risk.sort_values('mean', ascending=False)
    
    fig = go.Figure()
    
    # Add bars
    fig.add_trace(go.Bar(
        x=industry_risk['industry'],
        y=industry_risk['mean'],
        name='Average Risk',
        marker_color='rgba(55, 83, 109, 0.8)',
        text=industry_risk['mean'].round(3),
        textposition='outside'
    ))
    
    # Add error bars for standard deviation
    fig.add_trace(go.Scatter(
        x=industry_risk['industry'],
        y=industry_risk['mean'],
        error_y=dict(type='data', array=industry_risk['std'], visible=True),
        mode='markers',
        name='Risk Range',
        marker=dict(color='red', size=8)
    ))
    
    fig.update_layout(
        title="Average Risk Probability by Industry",
        xaxis_title="Industry",
        yaxis_title="Average Risk Probability",
        showlegend=True,
        template="plotly_white",
        height=500,
        xaxis_tickangle=-45
    )
    
    return fig


def create_financial_health_chart(df):
    """Create a comprehensive financial health radar chart."""
    
    # Select key financial metrics for radar chart
    metrics = ['leverage_ratio', 'current_ratio', 'interest_coverage', 'ebitda_margin', 'altman_z_score']
    
    # Normalize metrics for radar chart (0-1 scale)
    normalized_data = df[metrics].copy()
    for metric in metrics:
        if metric in ['leverage_ratio']:  # Lower is better
            normalized_data[metric] = 1 - (normalized_data[metric] - normalized_data[metric].min()) / (normalized_data[metric].max() - normalized_data[metric].min() + 1e-6)
        else:  # Higher is better
            normalized_data[metric] = (normalized_data[metric] - normalized_data[metric].min()) / (normalized_data[metric].max() - normalized_data[metric].min() + 1e-6)
    
    # Calculate averages by industry
    industry_avg = normalized_data.groupby(df['industry']).mean()
    
    fig = go.Figure()
    
    # Add radar chart for each industry (top 5)
    top_industries = df['industry'].value_counts().head(5).index
    
    for i, industry in enumerate(top_industries):
        values = industry_avg.loc[industry].values
        values = np.append(values, values[0])  # Close the radar
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=metrics,
            fill='toself',
            name=industry,
            opacity=0.7
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        title="Financial Health by Industry (Normalized)",
        template="plotly_white",
        height=600
    )
    
    return fig


def create_risk_scatter_chart(df):
    """Create a scatter plot showing relationship between risk and key metrics."""
    
    fig = go.Figure()
    
    # Create risk categories for coloring
    df['risk_category'] = pd.cut(
        df['predicted_proba'], 
        bins=[0, 0.3, 0.6, 1.0], 
        labels=['Low Risk', 'Medium Risk', 'High Risk']
    )
    
    colors = {'Low Risk': 'green', 'Medium Risk': 'orange', 'High Risk': 'red'}
    
    for category in df['risk_category'].unique():
        category_data = df[df['risk_category'] == category]
        
        fig.add_trace(go.Scatter(
            x=category_data['leverage_ratio'],
            y=category_data['predicted_proba'],
            mode='markers',
            name=category,
            marker=dict(
                color=colors.get(category, 'blue'),
                size=8,
                opacity=0.7
            ),
            text=category_data['company_id'],
            hovertemplate='<b>%{text}</b><br>Leverage: %{x:.3f}<br>Risk: %{y:.3f}<extra></extra>'
        ))
    
    fig.update_layout(
        title="Risk vs Leverage Ratio",
        xaxis_title="Leverage Ratio (Debt/Equity)",
        yaxis_title="Risk Probability",
        showlegend=True,
        template="plotly_white",
        height=500
    )
    
    return fig


def create_correlation_heatmap(df):
    """Create a clear correlation heatmap of key metrics."""
    
    # Select key metrics for correlation
    key_metrics = [
        'predicted_proba', 'leverage_ratio', 'current_ratio', 'interest_coverage',
        'ebitda_margin', 'altman_z_score', 'debt_to_assets', 'roa', 'roe'
    ]
    
    # Filter available columns
    available_metrics = [m for m in key_metrics if m in df.columns]
    corr_data = df[available_metrics].corr()
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_data.values,
        x=corr_data.columns,
        y=corr_data.columns,
        colorscale='RdBu',
        zmid=0,
        text=corr_data.round(2).values,
        texttemplate="%{text}",
        textfont={"size": 10},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title="Correlation Matrix: Key Risk Metrics",
        template="plotly_white",
        height=600,
        width=800
    )
    
    return fig


def create_portfolio_summary_cards(df):
    """Create summary cards for portfolio overview."""
    
    # Calculate key metrics
    total_companies = len(df)
    high_risk_companies = (df['predicted_proba'] > 0.6).sum()
    avg_risk = df['predicted_proba'].mean()
    industries = df['industry'].nunique()
    
    # Create columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Companies",
            f"{total_companies:,}",
            delta=None
        )
    
    with col2:
        st.metric(
            "High Risk Companies",
            f"{high_risk_companies:,}",
            delta=f"{high_risk_companies/total_companies:.1%}"
        )
    
    with col3:
        st.metric(
            "Average Risk Score",
            f"{avg_risk:.3f}",
            delta=None
        )
    
    with col4:
        st.metric(
            "Industries Covered",
            f"{industries}",
            delta=None
        )


def create_trend_chart(df):
    """Create a trend chart showing risk over time if date data available."""
    
    if 'as_of_date' not in df.columns:
        return None
    
    # Convert date and sort
    df['as_of_date'] = pd.to_datetime(df['as_of_date'])
    df_sorted = df.sort_values('as_of_date')
    
    # Calculate rolling average risk
    df_sorted['rolling_risk'] = df_sorted.groupby('as_of_date')['predicted_proba'].transform('mean')
    
    fig = go.Figure()
    
    # Add trend line
    fig.add_trace(go.Scatter(
        x=df_sorted['as_of_date'],
        y=df_sorted['rolling_risk'],
        mode='lines+markers',
        name='Average Risk Trend',
        line=dict(color='blue', width=3),
        marker=dict(size=8)
    ))
    
    # Add confidence bands
    fig.add_trace(go.Scatter(
        x=df_sorted['as_of_date'],
        y=df_sorted['rolling_risk'] + df_sorted['predicted_proba'].std(),
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        name='Upper Bound'
    ))
    
    fig.add_trace(go.Scatter(
        x=df_sorted['as_of_date'],
        y=df_sorted['rolling_risk'] - df_sorted['predicted_proba'].std(),
        mode='lines',
        line=dict(width=0),
        fill='tonexty',
        fillcolor='rgba(0,100,80,0.2)',
        showlegend=False,
        name='Lower Bound'
    ))
    
    fig.update_layout(
        title="Risk Trend Over Time",
        xaxis_title="Date",
        yaxis_title="Risk Probability",
        template="plotly_white",
        height=400
    )
    
    return fig
