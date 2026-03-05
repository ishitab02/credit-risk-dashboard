import streamlit as st
import plotly.express as px
from sqlalchemy import create_engine
from src.queries import (
    get_portfolio_kpis,
    get_default_by_gender,
    get_default_by_income_type,
    get_default_by_contract_type,
    get_income_distribution,
    get_credit_distribution
)

st.set_page_config(
    page_title="Portfolio Overview | Credit Risk",
    page_icon="PO",
    layout="wide"
)


@st.cache_resource
def get_db_engine():
    return create_engine(
        "sqlite:///database/credit_risk.db",
        connect_args={"check_same_thread": False}
    )


@st.cache_data(ttl=3600)
def load_portfolio_data(_engine):
    """Load all portfolio data with caching."""
    return {
        'kpis': get_portfolio_kpis(_engine),
        'gender': get_default_by_gender(_engine),
        'income_type': get_default_by_income_type(_engine),
        'contract': get_default_by_contract_type(_engine),
        'income_dist': get_income_distribution(_engine),
        'credit_dist': get_credit_distribution(_engine)
    }


# Get engine
engine = get_db_engine()

st.title("Portfolio Overview")
st.markdown("Key metrics and distributions for the loan portfolio")

# Load data
data = load_portfolio_data(engine)
kpis = data['kpis'].iloc[0]

# KPI Cards - Row 1
st.markdown("### Key Metrics")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="Total Borrowers",
        value=f"{kpis['total_borrowers']:,.0f}"
    )

with col2:
    default_rate = kpis['default_rate_pct']
    st.metric(
        label="Default Rate",
        value=f"{default_rate:.2f}%",
        delta=f"{'Above' if default_rate > 8 else 'Below'} 8% threshold",
        delta_color="inverse"
    )

with col3:
    st.metric(
        label="Avg Loan Amount",
        value=f"${kpis['avg_credit']:,.0f}"
    )

# KPI Cards - Row 2
col4, col5, col6 = st.columns(3)

with col4:
    st.metric(
        label="Avg Annual Income",
        value=f"${kpis['avg_income']:,.0f}"
    )

with col5:
    st.metric(
        label="Avg Annuity Payment",
        value=f"${kpis['avg_annuity']:,.0f}"
    )

with col6:
    st.metric(
        label="Avg Loan-to-Income",
        value=f"{kpis['avg_loan_to_income']:.2f}x"
    )

st.markdown("---")

# Charts Row 1
col_left, col_right = st.columns(2)

with col_left:
    st.markdown("### Income Distribution by Default Status")
    income_df = data['income_dist']
    income_df['Status'] = income_df['TARGET'].map({0: 'Repaid', 1: 'Defaulted'})
    fig_income = px.histogram(
        income_df,
        x='AMT_INCOME_TOTAL',
        color='Status',
        barmode='overlay',
        nbins=50,
        color_discrete_map={'Repaid': '#2ecc71', 'Defaulted': '#e74c3c'},
        template='plotly_dark',
        labels={'AMT_INCOME_TOTAL': 'Annual Income ($)'}
    )
    fig_income.update_layout(
        bargap=0.1,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_income, use_container_width=True)

with col_right:
    st.markdown("### Default Rate by Gender")
    gender_df = data['gender']
    fig_gender = px.bar(
        gender_df,
        x='CODE_GENDER',
        y='default_rate',
        color='default_rate',
        color_continuous_scale=['#2ecc71', '#f39c12', '#e74c3c'],
        template='plotly_dark',
        labels={'CODE_GENDER': 'Gender', 'default_rate': 'Default Rate (%)'},
        text='default_rate'
    )
    fig_gender.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig_gender.update_layout(showlegend=False, coloraxis_showscale=False)
    st.plotly_chart(fig_gender, use_container_width=True)

# Charts Row 2
col_left2, col_right2 = st.columns(2)

with col_left2:
    st.markdown("### Default Rate by Income Type")
    income_type_df = data['income_type']
    fig_income_type = px.bar(
        income_type_df,
        y='NAME_INCOME_TYPE',
        x='default_rate',
        orientation='h',
        color='default_rate',
        color_continuous_scale=['#2ecc71', '#f39c12', '#e74c3c'],
        template='plotly_dark',
        labels={'NAME_INCOME_TYPE': 'Income Type', 'default_rate': 'Default Rate (%)'},
        text='default_rate'
    )
    fig_income_type.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig_income_type.update_layout(showlegend=False, coloraxis_showscale=False, yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig_income_type, use_container_width=True)

with col_right2:
    st.markdown("### Loan Type Distribution")
    contract_df = data['contract']
    fig_contract = px.pie(
        contract_df,
        values='count',
        names='NAME_CONTRACT_TYPE',
        template='plotly_dark',
        color_discrete_sequence=['#1f4e79', '#3498db'],
        hole=0.4
    )
    fig_contract.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig_contract, use_container_width=True)

# Charts Row 3
st.markdown("### Loan Amount Distribution by Default Status")
credit_df = data['credit_dist']
credit_df['Status'] = credit_df['TARGET'].map({0: 'Repaid', 1: 'Defaulted'})
fig_credit = px.histogram(
    credit_df,
    x='AMT_CREDIT',
    color='Status',
    barmode='overlay',
    nbins=50,
    color_discrete_map={'Repaid': '#2ecc71', 'Defaulted': '#e74c3c'},
    template='plotly_dark',
    labels={'AMT_CREDIT': 'Loan Amount ($)'}
)
fig_credit.update_layout(
    bargap=0.1,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig_credit, use_container_width=True)
