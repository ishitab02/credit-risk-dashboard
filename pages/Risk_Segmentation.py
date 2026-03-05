import streamlit as st
import pandas as pd
import plotly.express as px
from sqlalchemy import create_engine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from src.queries import get_risk_features_for_clustering
from src.transforms import add_derived_features, impute_nulls, map_clusters_to_risk

st.set_page_config(
    page_title="Risk Segmentation | Credit Risk",
    page_icon="RS",
    layout="wide"
)


@st.cache_resource
def get_db_engine():
    return create_engine(
        "sqlite:///database/credit_risk.db",
        connect_args={"check_same_thread": False}
    )


@st.cache_data(ttl=3600)
def perform_clustering(_engine):
    """Performs K-Means clustering on borrower data."""
    # Load data
    df = get_risk_features_for_clustering(_engine)

    # Add derived features
    df = add_derived_features(df)
    df = impute_nulls(df)

    # Clustering features
    cluster_features = [
        'EXT_SOURCE_AVG',
        'DEBT_TO_INCOME',
        'ANNUITY_TO_INCOME',
        'AGE_YEARS',
        'EMPLOYED_YEARS'
    ]

    # Prepare data for clustering
    X_cluster = df[cluster_features].copy()

    # Handle any remaining NaN
    X_cluster = X_cluster.fillna(X_cluster.median())

    # Clip outliers for stability
    for col in ['DEBT_TO_INCOME', 'ANNUITY_TO_INCOME']:
        lo = X_cluster[col].quantile(0.01)
        hi = X_cluster[col].quantile(0.99)
        X_cluster[col] = X_cluster[col].clip(lo, hi)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)

    # K-Means clustering
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_scaled)

    # Map clusters to risk labels
    df, risk_mapping = map_clusters_to_risk(df, 'cluster', 'EXT_SOURCE_AVG')

    return df, cluster_features, risk_mapping


# Get engine
engine = get_db_engine()

st.title("Risk Segmentation")
st.markdown("K-Means clustering to segment borrowers into risk tiers")

# Perform clustering
with st.spinner("Running clustering analysis..."):
    df, features, risk_mapping = perform_clustering(engine)

# Summary stats
st.markdown("### Segment Summary")

tier_order = ['Low Risk', 'Medium Risk', 'High Risk']

summary = df.groupby('RISK_TIER').agg({
    'SK_ID_CURR': 'count',
    'TARGET': 'mean',
    'EXT_SOURCE_AVG': 'mean',
    'DEBT_TO_INCOME': 'mean',
    'AGE_YEARS': 'mean',
    'AMT_INCOME_TOTAL': 'mean'
}).reset_index()
summary.columns = ['Risk Tier', 'Count', 'Default Rate', 'Avg Bureau Score', 'Avg Debt/Income', 'Avg Age', 'Avg Income']
summary['Default Rate'] = (summary['Default Rate'] * 100).round(2).astype(str) + '%'
summary['Avg Bureau Score'] = summary['Avg Bureau Score'].round(3)
summary['Avg Debt/Income'] = summary['Avg Debt/Income'].round(2)
summary['Avg Age'] = summary['Avg Age'].round(1)
summary['Avg Income'] = summary['Avg Income'].apply(lambda x: f"${x:,.0f}")

# Reorder rows
summary['sort_key'] = summary['Risk Tier'].map({t: i for i, t in enumerate(tier_order)})
summary = summary.sort_values('sort_key').drop('sort_key', axis=1)

st.dataframe(summary, use_container_width=True, hide_index=True)

st.markdown("---")

# Charts
col1, col2 = st.columns(2)

with col1:
    st.markdown("### Risk Scatter Plot")
    st.caption("External Bureau Score vs Debt-to-Income Ratio")

    # Sample for performance
    plot_df = df.sample(min(5000, len(df)), random_state=42)

    fig_scatter = px.scatter(
        plot_df,
        x='EXT_SOURCE_AVG',
        y='DEBT_TO_INCOME',
        color='RISK_TIER',
        opacity=0.5,
        template='plotly_dark',
        color_discrete_map={
            'Low Risk': '#2ecc71',
            'Medium Risk': '#f39c12',
            'High Risk': '#e74c3c'
        },
        labels={
            'EXT_SOURCE_AVG': 'External Bureau Score (avg)',
            'DEBT_TO_INCOME': 'Debt-to-Income Ratio',
            'RISK_TIER': 'Risk Tier'
        }
    )
    fig_scatter.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

with col2:
    st.markdown("### Default Rate by Risk Tier")

    tier_default = df.groupby('RISK_TIER')['TARGET'].mean().reset_index()
    tier_default['Default Rate (%)'] = tier_default['TARGET'] * 100
    tier_default['sort_key'] = tier_default['RISK_TIER'].map({t: i for i, t in enumerate(tier_order)})
    tier_default = tier_default.sort_values('sort_key')

    fig_bar = px.bar(
        tier_default,
        x='RISK_TIER',
        y='Default Rate (%)',
        color='RISK_TIER',
        template='plotly_dark',
        color_discrete_map={
            'Low Risk': '#2ecc71',
            'Medium Risk': '#f39c12',
            'High Risk': '#e74c3c'
        },
        text='Default Rate (%)'
    )
    fig_bar.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig_bar.update_layout(showlegend=False, xaxis_title='Risk Tier', yaxis_title='Default Rate (%)')
    st.plotly_chart(fig_bar, use_container_width=True)

# Row 2
col3, col4 = st.columns(2)

with col3:
    st.markdown("### Income Distribution by Risk Tier")
    fig_box = px.box(
        df,
        x='RISK_TIER',
        y='AMT_INCOME_TOTAL',
        color='RISK_TIER',
        template='plotly_dark',
        color_discrete_map={
            'Low Risk': '#2ecc71',
            'Medium Risk': '#f39c12',
            'High Risk': '#e74c3c'
        },
        category_orders={'RISK_TIER': tier_order}
    )
    fig_box.update_layout(showlegend=False, xaxis_title='Risk Tier', yaxis_title='Annual Income ($)')
    fig_box.update_yaxes(range=[0, df['AMT_INCOME_TOTAL'].quantile(0.95)])
    st.plotly_chart(fig_box, use_container_width=True)

with col4:
    st.markdown("### Age Distribution by Risk Tier")
    fig_age = px.box(
        df,
        x='RISK_TIER',
        y='AGE_YEARS',
        color='RISK_TIER',
        template='plotly_dark',
        color_discrete_map={
            'Low Risk': '#2ecc71',
            'Medium Risk': '#f39c12',
            'High Risk': '#e74c3c'
        },
        category_orders={'RISK_TIER': tier_order}
    )
    fig_age.update_layout(showlegend=False, xaxis_title='Risk Tier', yaxis_title='Age (years)')
    st.plotly_chart(fig_age, use_container_width=True)

# Methodology explanation
st.markdown("---")
with st.expander("Methodology"):
    st.markdown("""
    **Clustering Approach:**
    - Algorithm: K-Means (k=3)
    - Features used: External Bureau Score Average, Debt-to-Income Ratio, Annuity-to-Income Ratio, Age, Employment Years
    - Sample size: 30,000 borrowers (for performance)

    **Risk Tier Assignment:**
    - Clusters are labeled based on average external bureau score
    - Higher bureau score → Lower risk tier
    - This ensures consistent labeling across runs

    **Key Insight:**
    The clustering reveals that external bureau scores are strong predictors of risk tier,
    with high-risk borrowers showing significantly higher default rates.
    """)
