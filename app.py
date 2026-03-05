import os
import streamlit as st
from sqlalchemy import create_engine


st.set_page_config(
    page_title="Credit Risk Dashboard",
    page_icon="CRD",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def get_db_engine():
    """Returns a cached SQLAlchemy engine."""
    return create_engine(
        "sqlite:///database/credit_risk.db",
        connect_args={"check_same_thread": False}
    )


def check_setup():
    """Ensures database and model are set up."""
    # Check and create database
    if not os.path.exists("database/credit_risk.db"):
        with st.spinner("Setting up database for first time (takes ~2 min)..."):
            from src.db_setup import setup_database
            setup_database()
            st.rerun()

    if not os.path.exists("models/default_scorer.pkl"):
        with st.spinner("Training model (takes ~1 min)..."):
            from models.train_model import train_models
            train_models()
            st.rerun()


# Ensure setup is complete
check_setup()

# Store engine in session state for pages to access
if 'engine' not in st.session_state:
    st.session_state.engine = get_db_engine()

# Home page content
st.title("Credit Risk Analytics Dashboard")
st.markdown("---")

st.markdown("""
### Welcome to the Credit Risk Dashboard

This dashboard analyzes **307,511 loan applications** to help identify borrowers at risk of default.

#### Navigate using the sidebar to explore:

| Page | Description |
|------|-------------|
| **Portfolio Overview** | Key metrics, distributions, and default rates by segment |
| **Risk Segmentation** | K-Means clustering of borrowers into risk tiers |
| **Default Drivers** | Correlation analysis to identify default predictors |
| **SQL Explorer** | Query the database directly with custom SQL |
| **Default Scorer** | Predict default probability for a borrower |

---

### Quick Stats
""")

# Show quick KPIs on home page
from src.queries import get_portfolio_kpis

engine = st.session_state.engine
kpis = get_portfolio_kpis(engine).iloc[0]

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Borrowers", f"{kpis['total_borrowers']:,.0f}")

with col2:
    st.metric("Default Rate", f"{kpis['default_rate_pct']:.2f}%")

with col3:
    st.metric("Avg Loan Amount", f"${kpis['avg_credit']:,.0f}")

with col4:
    st.metric("Avg Income", f"${kpis['avg_income']:,.0f}")

st.markdown("---")
st.caption("Built with Streamlit + SQLite + Scikit-learn | Data: Home Credit Default Risk")
