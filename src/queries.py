import re
import pandas as pd
from sqlalchemy import text


def get_portfolio_kpis(engine):
    """Returns portfolio-level KPI metrics."""
    sql = """
    SELECT
        COUNT(*) as total_borrowers,
        ROUND(100.0 * SUM(TARGET) / COUNT(*), 2) as default_rate_pct,
        ROUND(AVG(AMT_INCOME_TOTAL), 2) as avg_income,
        ROUND(AVG(AMT_CREDIT), 2) as avg_credit,
        ROUND(AVG(AMT_ANNUITY), 2) as avg_annuity,
        ROUND(AVG(AMT_CREDIT / (AMT_INCOME_TOTAL + 1)), 2) as avg_loan_to_income
    FROM applications
    """
    return pd.read_sql(sql, engine)


def get_default_by_gender(engine):
    """Returns default rate by gender."""
    sql = """
    SELECT
        CODE_GENDER,
        COUNT(*) as count,
        ROUND(100.0 * SUM(TARGET) / COUNT(*), 2) as default_rate
    FROM applications
    WHERE CODE_GENDER != 'XNA'
    GROUP BY CODE_GENDER
    """
    return pd.read_sql(sql, engine)


def get_default_by_income_type(engine):
    """Returns default rate by income type, sorted by default rate."""
    sql = """
    SELECT
        NAME_INCOME_TYPE,
        COUNT(*) as count,
        ROUND(100.0 * SUM(TARGET) / COUNT(*), 2) as default_rate
    FROM applications
    GROUP BY NAME_INCOME_TYPE
    ORDER BY default_rate DESC
    """
    return pd.read_sql(sql, engine)


def get_default_by_contract_type(engine):
    """Returns default rate by contract type."""
    sql = """
    SELECT
        NAME_CONTRACT_TYPE,
        COUNT(*) as count,
        ROUND(100.0 * SUM(TARGET) / COUNT(*), 2) as default_rate
    FROM applications
    GROUP BY NAME_CONTRACT_TYPE
    """
    return pd.read_sql(sql, engine)


def get_income_distribution(engine):
    """Returns income distribution capped at 1M for visualization."""
    sql = """
    SELECT AMT_INCOME_TOTAL, TARGET
    FROM applications
    WHERE AMT_INCOME_TOTAL <= 1000000
    LIMIT 50000
    """
    return pd.read_sql(sql, engine)


def get_credit_distribution(engine):
    """Returns credit amount distribution capped at 2M for visualization."""
    sql = """
    SELECT AMT_CREDIT, TARGET
    FROM applications
    WHERE AMT_CREDIT <= 2000000
    LIMIT 50000
    """
    return pd.read_sql(sql, engine)


def get_risk_features_for_clustering(engine):
    """Returns features needed for K-Means clustering (30k sample)."""
    sql = """
    SELECT
        SK_ID_CURR,
        TARGET,
        AMT_INCOME_TOTAL,
        AMT_CREDIT,
        AMT_ANNUITY,
        DAYS_BIRTH,
        DAYS_EMPLOYED,
        EXT_SOURCE_1,
        EXT_SOURCE_2,
        EXT_SOURCE_3
    FROM applications
    WHERE EXT_SOURCE_2 IS NOT NULL
    LIMIT 30000
    """
    return pd.read_sql(sql, engine)


def get_correlation_features(engine):
    """Returns numeric columns for correlation analysis (100k sample)."""
    sql = """
    SELECT
        TARGET,
        AMT_INCOME_TOTAL,
        AMT_CREDIT,
        AMT_ANNUITY,
        DAYS_BIRTH,
        DAYS_EMPLOYED,
        EXT_SOURCE_1,
        EXT_SOURCE_2,
        EXT_SOURCE_3,
        REGION_POPULATION_RELATIVE,
        OBS_30_CNT_SOCIAL_CIRCLE,
        DEF_30_CNT_SOCIAL_CIRCLE,
        AMT_REQ_CREDIT_BUREAU_YEAR
    FROM applications
    LIMIT 100000
    """
    return pd.read_sql(sql, engine)


def get_feature_distribution_by_target(engine, feature: str):
    """Returns feature distribution split by target for comparison."""
    # Whitelist allowed features to prevent SQL injection
    allowed_features = [
        'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3',
        'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY',
        'DAYS_BIRTH', 'DAYS_EMPLOYED',
        'REGION_POPULATION_RELATIVE',
        'OBS_30_CNT_SOCIAL_CIRCLE', 'DEF_30_CNT_SOCIAL_CIRCLE',
        'AMT_REQ_CREDIT_BUREAU_YEAR'
    ]
    if feature not in allowed_features:
        raise ValueError(f"Feature '{feature}' not allowed")

    sql = f"""
    SELECT TARGET, {feature}
    FROM applications
    WHERE {feature} IS NOT NULL
    LIMIT 50000
    """
    return pd.read_sql(sql, engine)


def get_model_training_data(engine):
    """Returns data for model training."""
    sql = """
    SELECT
        SK_ID_CURR,
        TARGET,
        AMT_INCOME_TOTAL,
        AMT_CREDIT,
        AMT_ANNUITY,
        DAYS_BIRTH,
        DAYS_EMPLOYED,
        EXT_SOURCE_1,
        EXT_SOURCE_2,
        EXT_SOURCE_3
    FROM applications
    WHERE TARGET IS NOT NULL
    """
    return pd.read_sql(sql, engine)


def run_custom_query(engine, sql: str):
    """
    Executes a custom SQL query with safety checks.
    Returns (DataFrame, error_message) tuple.
    """
    # Block dangerous SQL commands (case-insensitive)
    dangerous_patterns = [
        r'\bDROP\b', r'\bDELETE\b', r'\bINSERT\b',
        r'\bUPDATE\b', r'\bALTER\b', r'\bCREATE\b',
        r'\bTRUNCATE\b', r'\bEXEC\b', r'\bEXECUTE\b'
    ]

    for pattern in dangerous_patterns:
        if re.search(pattern, sql, re.IGNORECASE):
            return None, f"Blocked: {pattern.strip(chr(92)).strip('b')} statements not allowed"

    try:
        df = pd.read_sql(sql, engine)
        return df, None
    except Exception as e:
        return None, str(e)


# Schema reference for SQL Explorer
SCHEMA_REFERENCE = """
TABLE: applications (307,511 rows)
-------------------------------
SK_ID_CURR          - Unique borrower ID (primary key)
TARGET              - 1 = defaulted, 0 = repaid
AMT_INCOME_TOTAL    - Annual income
AMT_CREDIT          - Loan amount
AMT_ANNUITY         - Annual repayment
AMT_GOODS_PRICE     - Price of goods financed
DAYS_BIRTH          - Age in days (negative)
DAYS_EMPLOYED       - Employment duration (negative, 365243 = unemployed)
NAME_CONTRACT_TYPE  - Cash loans / Revolving loans
CODE_GENDER         - M / F / XNA
FLAG_OWN_CAR        - Y / N
FLAG_OWN_REALTY     - Y / N
NAME_INCOME_TYPE    - Working, Pensioner, etc.
NAME_EDUCATION_TYPE - Education level
NAME_FAMILY_STATUS  - Marital status
NAME_HOUSING_TYPE   - Housing type
OCCUPATION_TYPE     - Job type
EXT_SOURCE_1/2/3    - External bureau scores (0-1, higher = better)

TABLE: bureau
-------------
SK_ID_CURR          - FK to applications
SK_ID_BUREAU        - Unique bureau record ID
CREDIT_ACTIVE       - Active / Closed / Sold / Bad debt
CREDIT_TYPE         - Type of external credit
AMT_CREDIT_SUM      - Credit amount
AMT_CREDIT_SUM_DEBT - Current debt outstanding
AMT_CREDIT_SUM_OVERDUE - Overdue amount
CREDIT_DAY_OVERDUE  - Days overdue

TABLE: previous_application
---------------------------
SK_ID_CURR          - FK to applications
SK_ID_PREV          - Unique previous app ID
NAME_CONTRACT_STATUS- Approved / Refused / Canceled / Unused offer
AMT_APPLICATION     - Requested amount
AMT_CREDIT          - Approved credit amount
NAME_CONTRACT_TYPE  - Type of previous loan
"""


EXAMPLE_QUERIES = {
    "Portfolio Default Rate": """
SELECT
    COUNT(*) as total_borrowers,
    SUM(TARGET) as total_defaults,
    ROUND(100.0 * SUM(TARGET) / COUNT(*), 2) as default_rate_pct
FROM applications
""",
    "Default Rate by Gender": """
SELECT
    CODE_GENDER,
    COUNT(*) as count,
    ROUND(100.0 * SUM(TARGET)/COUNT(*), 2) as default_rate_pct
FROM applications
WHERE CODE_GENDER != 'XNA'
GROUP BY CODE_GENDER
ORDER BY default_rate_pct DESC
""",
    "Top 10 Income Types by Default Rate": """
SELECT
    NAME_INCOME_TYPE,
    COUNT(*) as borrower_count,
    ROUND(AVG(AMT_INCOME_TOTAL), 0) as avg_income,
    ROUND(100.0 * SUM(TARGET)/COUNT(*), 2) as default_rate_pct
FROM applications
GROUP BY NAME_INCOME_TYPE
HAVING COUNT(*) > 100
ORDER BY default_rate_pct DESC
LIMIT 10
""",
    "Bureau Data JOIN": """
SELECT
    a.CODE_GENDER,
    COUNT(DISTINCT a.SK_ID_CURR) as borrowers,
    ROUND(AVG(b.loan_count), 1) as avg_bureau_loans,
    ROUND(100.0 * SUM(a.TARGET)/COUNT(*), 2) as default_rate_pct
FROM applications a
LEFT JOIN (
    SELECT SK_ID_CURR, COUNT(*) as loan_count
    FROM bureau
    GROUP BY SK_ID_CURR
) b ON a.SK_ID_CURR = b.SK_ID_CURR
WHERE a.CODE_GENDER != 'XNA'
GROUP BY a.CODE_GENDER
""",
    "Previous Applications Analysis": """
SELECT
    p.NAME_CONTRACT_STATUS,
    COUNT(*) as count,
    ROUND(AVG(a.AMT_INCOME_TOTAL), 0) as avg_income,
    ROUND(100.0 * SUM(a.TARGET)/COUNT(*), 2) as default_rate_pct
FROM applications a
JOIN previous_application p ON a.SK_ID_CURR = p.SK_ID_CURR
GROUP BY p.NAME_CONTRACT_STATUS
ORDER BY default_rate_pct DESC
""",
    "Age Analysis (Derived)": """
SELECT
    CASE
        WHEN DAYS_BIRTH / -365 < 25 THEN '18-24'
        WHEN DAYS_BIRTH / -365 < 35 THEN '25-34'
        WHEN DAYS_BIRTH / -365 < 45 THEN '35-44'
        WHEN DAYS_BIRTH / -365 < 55 THEN '45-54'
        ELSE '55+'
    END as age_group,
    COUNT(*) as borrower_count,
    ROUND(100.0 * SUM(TARGET) / COUNT(*), 2) as default_rate_pct
FROM applications
GROUP BY age_group
ORDER BY age_group
"""
}
