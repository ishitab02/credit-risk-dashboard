import streamlit as st
from sqlalchemy import create_engine
from src.queries import run_custom_query, SCHEMA_REFERENCE, EXAMPLE_QUERIES

st.set_page_config(
    page_title="SQL Explorer | Credit Risk",
    page_icon="SQL",
    layout="wide"
)


@st.cache_resource
def get_db_engine():
    return create_engine(
        "sqlite:///database/credit_risk.db",
        connect_args={"check_same_thread": False}
    )


# Get engine
engine = get_db_engine()

st.title("SQL Query Explorer")
st.markdown("Query the credit risk database directly")
st.caption("Tables: `applications`, `bureau`, `previous_application`")

# Schema reference
with st.expander("Show Table Schemas"):
    st.code(SCHEMA_REFERENCE, language='text')

st.markdown("---")

# Pre-built example queries
example_names = ["-- Select an example --"] + list(EXAMPLE_QUERIES.keys())
selected_example = st.selectbox(
    "Load an example query",
    example_names,
    index=0
)

# Set initial query
if selected_example != "-- Select an example --":
    initial_query = EXAMPLE_QUERIES[selected_example]
else:
    initial_query = """SELECT
    COUNT(*) as total_borrowers,
    ROUND(100.0 * SUM(TARGET) / COUNT(*), 2) as default_rate_pct
FROM applications"""

# Query input
query_text = st.text_area(
    "SQL Query",
    value=initial_query,
    height=180,
    help="Write SELECT queries only. DROP, DELETE, INSERT, UPDATE are blocked."
)

# Execute button
col1, col2 = st.columns([1, 5])
with col1:
    run_button = st.button("Run Query", type="primary", use_container_width=True)

if run_button:
    if not query_text.strip():
        st.warning("Please enter a SQL query")
    else:
        with st.spinner("Executing query..."):
            df, error = run_custom_query(engine, query_text)

        if error:
            st.error(f"Query Error: {error}")
        else:
            st.success(f"{len(df):,} rows returned")

            # Display results
            st.dataframe(df, use_container_width=True, hide_index=True)

            # Download button
            csv_data = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name="query_result.csv",
                mime="text/csv"
            )

            # Quick stats for numeric columns
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0 and len(df) > 1:
                with st.expander("Quick Stats"):
                    st.dataframe(df[numeric_cols].describe(), use_container_width=True)

st.markdown("---")

# Tips section
with st.expander("Query Tips"):
    st.markdown("""
    **Common Patterns:**

    ```sql
    -- Default rate by any categorical column
    SELECT column_name,
           COUNT(*) as count,
           ROUND(100.0 * SUM(TARGET) / COUNT(*), 2) as default_rate_pct
    FROM applications
    GROUP BY column_name
    ORDER BY default_rate_pct DESC
    ```

    ```sql
    -- Age analysis (DAYS_BIRTH is negative)
    SELECT DAYS_BIRTH / -365 as age_years, TARGET
    FROM applications
    LIMIT 1000
    ```

    ```sql
    -- Join with bureau data
    SELECT a.SK_ID_CURR, a.TARGET,
           COUNT(b.SK_ID_BUREAU) as bureau_count
    FROM applications a
    LEFT JOIN bureau b ON a.SK_ID_CURR = b.SK_ID_CURR
    GROUP BY a.SK_ID_CURR
    LIMIT 100
    ```

    **Notes:**
    - DAYS columns are negative (days before application)
    - DAYS_EMPLOYED = 365243 means unemployed
    - EXT_SOURCE columns range 0-1 (higher = better)
    - Use LIMIT for large result sets
    """)
