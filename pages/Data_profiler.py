import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine

st.set_page_config(page_title="Data Profiler", page_icon="DPR", layout="wide")


@st.cache_resource
def get_engine():
    return create_engine(
        "sqlite:///database/credit_risk.db",
        connect_args={"check_same_thread": False},
    )


@st.cache_data(ttl=3600)
def get_table_names(_engine):
    df = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table' AND name != 'pipeline_runs'", _engine)
    return df["name"].tolist()


@st.cache_data(ttl=3600)
def get_row_count(_engine, table):
    df = pd.read_sql(f"SELECT COUNT(*) as cnt FROM {table}", _engine)
    return int(df["cnt"].iloc[0])


@st.cache_data(ttl=3600)
def get_sample(_engine, table, limit=50000):
    return pd.read_sql(f"SELECT * FROM {table} LIMIT {limit}", _engine)


@st.cache_data(ttl=3600)
def compute_profile(_engine, table):
    """Compute column-level profiling stats."""
    df = get_sample(_engine, table)
    total = len(df)
    profiles = []

    for col in df.columns:
        series = df[col]
        null_count = int(series.isnull().sum())
        null_pct = null_count / total if total > 0 else 0
        unique_count = int(series.nunique())
        dtype = str(series.dtype)

        profile = {
            "Column": col,
            "Type": dtype,
            "Non-Null": f"{total - null_count:,}",
            "Null %": f"{null_pct:.1%}",
            "Unique": f"{unique_count:,}",
            "Cardinality": f"{unique_count / total:.2%}" if total > 0 else "0%",
        }

        if pd.api.types.is_numeric_dtype(series):
            clean = series.dropna()
            if len(clean) > 0:
                profile["Min"] = f"{clean.min():,.2f}"
                profile["Max"] = f"{clean.max():,.2f}"
                profile["Mean"] = f"{clean.mean():,.2f}"
                profile["Median"] = f"{clean.median():,.2f}"
                profile["Std Dev"] = f"{clean.std():,.2f}"
            else:
                profile.update({"Min": "-", "Max": "-", "Mean": "-", "Median": "-", "Std Dev": "-"})
        else:
            top_val = series.mode().iloc[0] if len(series.mode()) > 0 else "-"
            profile.update({"Min": "-", "Max": "-", "Mean": "-", "Median": "-", "Std Dev": str(top_val)[:30]})
            profile["Std Dev"] = f"mode: {top_val}"

        profiles.append(profile)

    return pd.DataFrame(profiles), df


engine = get_engine()

st.title("Data Profiler")
st.markdown("Column-level statistics, null patterns, and distribution analysis.")
st.markdown("---")

tables = get_table_names(engine)
if not tables:
    st.warning("No tables found. Run the ETL pipeline first.")
    st.stop()

selected_table = st.selectbox("Select Table", tables)
row_count = get_row_count(engine, selected_table)

profile_df, sample_df = compute_profile(engine, selected_table)
num_cols = len(sample_df.columns)
num_numeric = sum(1 for c in sample_df.columns if pd.api.types.is_numeric_dtype(sample_df[c]))
num_categorical = num_cols - num_numeric
total_nulls = sample_df.isnull().sum().sum()
total_cells = sample_df.shape[0] * sample_df.shape[1]
overall_completeness = 1 - (total_nulls / total_cells) if total_cells > 0 else 1

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Rows", f"{row_count:,}")
k2.metric("Columns", num_cols)
k3.metric("Numeric", num_numeric)
k4.metric("Categorical", num_categorical)
k5.metric("Completeness", f"{overall_completeness:.1%}")

st.markdown("---")
st.subheader("Column Statistics")
st.dataframe(profile_df, use_container_width=True, hide_index=True, height=400)

st.markdown("---")
st.subheader("Null Pattern Analysis")

null_rates = sample_df.isnull().mean().sort_values(ascending=False)
cols_with_nulls = null_rates[null_rates > 0]

if len(cols_with_nulls) > 0:
    col_left, col_right = st.columns([2, 1])

    with col_left:
        fig_null = px.bar(
            x=cols_with_nulls.values * 100,
            y=cols_with_nulls.index,
            orientation="h",
            labels={"x": "Null Rate (%)", "y": "Column"},
            title="Null Rate by Column",
            template="plotly_dark",
            color=cols_with_nulls.values * 100,
            color_continuous_scale=["#2ecc71", "#f39c12", "#e74c3c"],
        )
        fig_null.update_layout(showlegend=False, height=max(300, len(cols_with_nulls) * 30))
        st.plotly_chart(fig_null, use_container_width=True)

    with col_right:
        st.markdown("**Null Summary**")
        null_summary = pd.DataFrame({
            "Column": cols_with_nulls.index,
            "Null Rate": [f"{v:.1%}" for v in cols_with_nulls.values],
            "Null Count": [f"{int(sample_df[c].isnull().sum()):,}" for c in cols_with_nulls.index],
        })
        st.dataframe(null_summary, hide_index=True)
else:
    st.success("No null values found in this table.")

st.markdown("---")
st.subheader("Distribution Explorer")

numeric_cols = [c for c in sample_df.columns if pd.api.types.is_numeric_dtype(sample_df[c])]
categorical_cols = [c for c in sample_df.columns if not pd.api.types.is_numeric_dtype(sample_df[c])]

tab_num, tab_cat = st.tabs(["Numeric Distributions", "Categorical Distributions"])

with tab_num:
    if numeric_cols:
        selected_numeric = st.selectbox("Select numeric column", numeric_cols, key="num_col")
        clean_data = sample_df[selected_numeric].dropna()

        if len(clean_data) > 0:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Mean", f"{clean_data.mean():,.2f}")
            c2.metric("Median", f"{clean_data.median():,.2f}")
            c3.metric("Std Dev", f"{clean_data.std():,.2f}")
            c4.metric("Skewness", f"{clean_data.skew():.2f}")

            fig_hist = px.histogram(
                clean_data, nbins=50,
                title=f"Distribution of {selected_numeric}",
                template="plotly_dark",
                color_discrete_sequence=["#1f4e79"],
            )
            fig_hist.update_layout(xaxis_title=selected_numeric, yaxis_title="Count", showlegend=False)
            st.plotly_chart(fig_hist, use_container_width=True)

            fig_box = px.box(
                sample_df, y=selected_numeric,
                title=f"Box Plot of {selected_numeric}",
                template="plotly_dark",
                color_discrete_sequence=["#1f4e79"],
            )
            st.plotly_chart(fig_box, use_container_width=True)
    else:
        st.info("No numeric columns in this table.")

with tab_cat:
    if categorical_cols:
        selected_cat = st.selectbox("Select categorical column", categorical_cols, key="cat_col")
        value_counts = sample_df[selected_cat].value_counts().head(20)

        c1, c2 = st.columns(2)
        c1.metric("Unique Values", f"{sample_df[selected_cat].nunique():,}")
        c2.metric("Most Common", str(value_counts.index[0]) if len(value_counts) > 0 else "-")

        fig_bar = px.bar(
            x=value_counts.index.astype(str),
            y=value_counts.values,
            title=f"Top {min(20, len(value_counts))} Values — {selected_cat}",
            template="plotly_dark",
            labels={"x": selected_cat, "y": "Count"},
            color_discrete_sequence=["#1f4e79"],
        )
        fig_bar.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("No categorical columns in this table.")

if selected_table == "applications" and len(numeric_cols) >= 2:
    st.markdown("---")
    st.subheader("Correlation Matrix")

    key_numeric = [c for c in [
        "TARGET", "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY",
        "DAYS_BIRTH", "DAYS_EMPLOYED", "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
    ] if c in numeric_cols]

    if len(key_numeric) >= 2:
        corr = sample_df[key_numeric].corr()
        fig_corr = px.imshow(
            corr,
            text_auto=".2f",
            title="Correlation Matrix (Key Features)",
            template="plotly_dark",
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
        )
        fig_corr.update_layout(height=500)
        st.plotly_chart(fig_corr, use_container_width=True)

st.markdown("---")
st.caption("Data Profiler — automated column statistics, null analysis, and distribution visualization")