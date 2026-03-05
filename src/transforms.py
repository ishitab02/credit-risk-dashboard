import pandas as pd
import numpy as np


def add_derived_features(df):
    """
    Adds derived features to the dataframe.

    Derived columns:
    - AGE_YEARS: Age in years (from DAYS_BIRTH)
    - EMPLOYED_YEARS: Employment duration in years (from DAYS_EMPLOYED)
    - DEBT_TO_INCOME: Loan amount / annual income
    - ANNUITY_TO_INCOME: Annual repayment / annual income
    - CREDIT_UTIL: Annual payment / loan amount
    - EXT_SOURCE_AVG: Average of external bureau scores
    """
    df = df.copy()

    # Age in years (DAYS_BIRTH is negative)
    if 'DAYS_BIRTH' in df.columns:
        df['AGE_YEARS'] = df['DAYS_BIRTH'] / -365

    # Employment years (DAYS_EMPLOYED is negative, 365243 means unemployed)
    if 'DAYS_EMPLOYED' in df.columns:
        df['EMPLOYED_YEARS'] = df['DAYS_EMPLOYED'].apply(
            lambda x: x / -365 if pd.notna(x) and x < 0 else 0
        )

    # Debt ratios
    if 'AMT_CREDIT' in df.columns and 'AMT_INCOME_TOTAL' in df.columns:
        df['DEBT_TO_INCOME'] = df['AMT_CREDIT'] / (df['AMT_INCOME_TOTAL'] + 1)

    if 'AMT_ANNUITY' in df.columns and 'AMT_INCOME_TOTAL' in df.columns:
        df['ANNUITY_TO_INCOME'] = df['AMT_ANNUITY'] / (df['AMT_INCOME_TOTAL'] + 1)

    if 'AMT_ANNUITY' in df.columns and 'AMT_CREDIT' in df.columns:
        df['CREDIT_UTIL'] = df['AMT_ANNUITY'] / (df['AMT_CREDIT'] + 1)

    # Average external source score
    ext_cols = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
    available_ext_cols = [c for c in ext_cols if c in df.columns]
    if available_ext_cols:
        df['EXT_SOURCE_AVG'] = df[available_ext_cols].mean(axis=1)

    return df


def clip_outliers(df, col, lower_pct=0.01, upper_pct=0.99):
    """
    Clips outliers in a column to specified percentiles.
    """
    df = df.copy()
    if col in df.columns:
        lo = df[col].quantile(lower_pct)
        hi = df[col].quantile(upper_pct)
        df[col] = df[col].clip(lo, hi)
    return df


def impute_nulls(df):
    """
    Imputes null values:
    - EXT_SOURCE columns: median
    - Other numerics: 0
    """
    df = df.copy()

    # Impute EXT_SOURCE with median
    for col in ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']:
        if col in df.columns:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

    # Impute other numerics with 0
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col not in ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']:
            df[col] = df[col].fillna(0)

    return df


def prepare_clustering_data(df):
    """
    Prepares data for K-Means clustering.
    Returns cleaned dataframe with clustering features.
    """
    df = add_derived_features(df)
    df = impute_nulls(df)

    # Features for clustering
    cluster_features = [
        'EXT_SOURCE_AVG',
        'DEBT_TO_INCOME',
        'ANNUITY_TO_INCOME',
        'AGE_YEARS',
        'EMPLOYED_YEARS'
    ]

    # Ensure all features exist
    available_features = [f for f in cluster_features if f in df.columns]

    # Clip outliers for clustering stability
    for col in ['DEBT_TO_INCOME', 'ANNUITY_TO_INCOME']:
        if col in df.columns:
            df = clip_outliers(df, col)

    return df, available_features


def map_clusters_to_risk(df, cluster_col='cluster', ext_source_col='EXT_SOURCE_AVG'):
    """
    Maps K-Means cluster numbers to meaningful risk labels.
    Higher EXT_SOURCE_AVG = Lower Risk.
    """
    df = df.copy()

    # Calculate mean EXT_SOURCE_AVG per cluster
    cluster_means = df.groupby(cluster_col)[ext_source_col].mean().sort_values(ascending=False)

    # Map: highest avg = Low Risk, lowest = High Risk
    risk_mapping = {}
    risk_labels = ['Low Risk', 'Medium Risk', 'High Risk']

    for i, cluster_id in enumerate(cluster_means.index):
        risk_mapping[cluster_id] = risk_labels[min(i, len(risk_labels) - 1)]

    df['RISK_TIER'] = df[cluster_col].map(risk_mapping)

    return df, risk_mapping


def prepare_model_features(df):
    """
    Prepares features for the default prediction model.
    Returns X (features) and y (target).
    """
    df = add_derived_features(df)
    df = impute_nulls(df)

    # Model features
    features = [
        'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY',
        'DAYS_BIRTH', 'DAYS_EMPLOYED',
        'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3',
        'DEBT_TO_INCOME', 'ANNUITY_TO_INCOME'
    ]

    # Filter to available features
    available = [f for f in features if f in df.columns]

    # Handle DAYS_EMPLOYED sentinel value (365243 = unemployed)
    if 'DAYS_EMPLOYED' in df.columns:
        df.loc[df['DAYS_EMPLOYED'] > 0, 'DAYS_EMPLOYED'] = 0

    X = df[available]
    y = df['TARGET'] if 'TARGET' in df.columns else None

    return X, y, available
