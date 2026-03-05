import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sqlalchemy import create_engine

from src.queries import get_correlation_features, get_feature_distribution_by_target
from src.transforms import add_derived_features

st.set_page_config(
    page_title="Default Drivers | Credit Risk",
    page_icon="DD",
    layout="wide"
)


@st.cache_resource
def get_db_engine():
    return create_engine(
        "sqlite:///database/credit_risk.db",
        connect_args={"check_same_thread": False}
    )


@st.cache_data(ttl=3600)
def load_correlation_data(_engine):
    """Loads and prepares data for correlation analysis."""
    df = get_correlation_features(_engine)
    df = add_derived_features(df)
    return df


@st.cache_data(ttl=3600)
def load_feature_distribution(_engine, feature):
    """Loads feature distribution by target."""
    df = get_feature_distribution_by_target(_engine, feature)
    return df


# Get engine
engine = get_db_engine()

st.title("Default Drivers")
st.markdown("Identifying features that predict loan default")

# Load correlation data
df = load_correlation_data(engine)

# Section 1: Correlation Bar Chart
st.markdown("### Feature Correlation with Default")

# Calculate correlations
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if 'TARGET' in numeric_cols:
    correlations = df[numeric_cols].corr()['TARGET'].drop('TARGET')
    corr_df = pd.DataFrame({
        'Feature': correlations.index,
        'Correlation': correlations.values
    })
    corr_df['Abs_Correlation'] = corr_df['Correlation'].abs()
    corr_df = corr_df.sort_values('Abs_Correlation', ascending=True).tail(15)

    # Color by sign
    corr_df['Direction'] = corr_df['Correlation'].apply(
        lambda x: 'Increases Default' if x > 0 else 'Decreases Default'
    )

    fig_corr = px.bar(
        corr_df,
        y='Feature',
        x='Correlation',
        orientation='h',
        color='Direction',
        color_discrete_map={
            'Increases Default': '#e74c3c',
            'Decreases Default': '#2ecc71'
        },
        template='plotly_dark',
        labels={'Correlation': 'Correlation with Default'}
    )
    fig_corr.update_layout(
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    st.info("""
    **Key Insight:** EXT_SOURCE_2 and EXT_SOURCE_3 (external bureau scores) show the strongest
    negative correlation with default — higher bureau scores strongly reduce default risk.
    DAYS_BIRTH also shows negative correlation, meaning older borrowers default less.
    """)

st.markdown("---")

# Section 2: Feature Distribution Comparison
st.markdown("### Feature Distribution: Defaulters vs Non-Defaulters")

feature_options = [
    'EXT_SOURCE_2', 'EXT_SOURCE_3', 'EXT_SOURCE_1',
    'DAYS_BIRTH', 'DAYS_EMPLOYED',
    'AMT_CREDIT', 'AMT_INCOME_TOTAL', 'AMT_ANNUITY'
]

col1, col2 = st.columns([1, 3])
with col1:
    selected_feature = st.selectbox(
        "Select Feature",
        feature_options,
        index=0
    )

# Load feature distribution
feature_df = load_feature_distribution(engine, selected_feature)
feature_df['Status'] = feature_df['TARGET'].map({0: 'Repaid', 1: 'Defaulted'})

fig_dist = px.histogram(
    feature_df,
    x=selected_feature,
    color='Status',
    barmode='overlay',
    nbins=50,
    histnorm='probability density',
    color_discrete_map={'Repaid': '#2ecc71', 'Defaulted': '#e74c3c'},
    template='plotly_dark',
    labels={selected_feature: selected_feature.replace('_', ' ').title()}
)
fig_dist.update_layout(
    bargap=0.1,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig_dist, use_container_width=True)

st.markdown("---")

# Section 3: Correlation Heatmap
st.markdown("### Correlation Heatmap")

# Select key features for heatmap
heatmap_features = [
    'TARGET', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3',
    'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY',
    'DAYS_BIRTH', 'DAYS_EMPLOYED'
]
heatmap_cols = [c for c in heatmap_features if c in df.columns]
corr_matrix = df[heatmap_cols].corr()

fig_heatmap = px.imshow(
    corr_matrix,
    text_auto='.2f',
    color_continuous_scale='RdBu_r',
    template='plotly_dark',
    aspect='auto'
)
fig_heatmap.update_layout(
    height=500,
    xaxis_title='',
    yaxis_title=''
)
st.plotly_chart(fig_heatmap, use_container_width=True)

# Additional insights
st.markdown("---")
st.markdown("### Key Takeaways")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    **Strongest Predictors**
    - External bureau scores (EXT_SOURCE_2, EXT_SOURCE_3)
    - Age (DAYS_BIRTH)
    - Employment duration
    """)

with col2:
    st.markdown("""
    **Risk Reducers**
    - Higher external scores
    - Older age
    - Longer employment
    - Lower debt-to-income
    """)

with col3:
    st.markdown("""
    **Risk Increasers**
    - Lower bureau scores
    - Higher credit bureau inquiries
    - Social circle defaults
    """)
