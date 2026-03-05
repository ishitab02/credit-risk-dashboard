import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
import os
import shap
from sqlalchemy import create_engine

st.set_page_config(
    page_title="Explainability | Credit Risk",
    page_icon="EX",
    layout="wide"
)

# Colors
GREEN = "#2ecc71"
RED = "#e74c3c"
ORANGE = "#f39c12"
BLUE = "#3498db"

# Feature names for display
FEATURE_NAMES = [
    'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY',
    'DAYS_BIRTH', 'DAYS_EMPLOYED',
    'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3',
    'DEBT_TO_INCOME', 'ANNUITY_TO_INCOME'
]

FEATURE_DISPLAY_NAMES = {
    'AMT_INCOME_TOTAL': 'Annual Income',
    'AMT_CREDIT': 'Loan Amount',
    'AMT_ANNUITY': 'Annual Payment',
    'DAYS_BIRTH': 'Age (days)',
    'DAYS_EMPLOYED': 'Employment (days)',
    'EXT_SOURCE_1': 'Bureau Score 1',
    'EXT_SOURCE_2': 'Bureau Score 2',
    'EXT_SOURCE_3': 'Bureau Score 3',
    'DEBT_TO_INCOME': 'Debt-to-Income',
    'ANNUITY_TO_INCOME': 'Annuity-to-Income'
}


@st.cache_resource
def load_model():
    """Load the trained logistic regression model."""
    model_path = "models/logreg_model.pkl"
    if not os.path.exists(model_path):
        model_path = "models/default_scorer.pkl"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None


@st.cache_resource
def get_shap_explainer(_model):
    """Create SHAP LinearExplainer for the logistic regression model."""
    # Extract classifier from pipeline
    classifier = _model.named_steps['classifier']

    # Get background data for SHAP (use training data statistics)
    # For LinearExplainer, we need the scaled data
    # We'll create a simple background with zeros (mean after scaling)
    background = np.zeros((1, len(FEATURE_NAMES)))

    explainer = shap.LinearExplainer(classifier, background)
    return explainer


@st.cache_data
def load_sample_data():
    """Load a sample of borrowers for global SHAP analysis."""
    db_path = "database/credit_risk.db"
    if not os.path.exists(db_path):
        return None

    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False}
    )

    sql = """
    SELECT
        AMT_INCOME_TOTAL, AMT_CREDIT, AMT_ANNUITY,
        DAYS_BIRTH, DAYS_EMPLOYED,
        EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3
    FROM applications
    WHERE EXT_SOURCE_1 IS NOT NULL
      AND EXT_SOURCE_2 IS NOT NULL
      AND EXT_SOURCE_3 IS NOT NULL
    LIMIT 1000
    """
    df = pd.read_sql(sql, engine)

    # Add derived features
    df['DEBT_TO_INCOME'] = df['AMT_CREDIT'] / (df['AMT_INCOME_TOTAL'] + 1)
    df['ANNUITY_TO_INCOME'] = df['AMT_ANNUITY'] / (df['AMT_INCOME_TOTAL'] + 1)

    # Handle DAYS_EMPLOYED sentinel
    df.loc[df['DAYS_EMPLOYED'] > 0, 'DAYS_EMPLOYED'] = 0

    return df[FEATURE_NAMES]


@st.cache_data
def compute_global_shap_values(_model, sample_df):
    """Compute SHAP values for a sample of borrowers."""
    # Transform through pipeline
    imputer = _model.named_steps['imputer']
    scaler = _model.named_steps['scaler']
    classifier = _model.named_steps['classifier']

    X_imputed = imputer.transform(sample_df)
    X_scaled = scaler.transform(X_imputed)

    # Create explainer and compute values
    explainer = shap.LinearExplainer(classifier, X_scaled)
    shap_values = explainer.shap_values(X_scaled)

    return shap_values, X_scaled


def get_risk_label(prob):
    """Returns risk label and color based on probability."""
    if prob < 0.15:
        return "Low Risk", GREEN
    elif prob < 0.30:
        return "Medium Risk", ORANGE
    else:
        return "High Risk", RED


def create_waterfall_plot(shap_values, feature_values, feature_names, base_value):
    """Create a Plotly waterfall chart for SHAP values."""
    # Sort by absolute SHAP value
    sorted_idx = np.argsort(np.abs(shap_values))[::-1]

    sorted_shap = shap_values[sorted_idx]
    sorted_names = [FEATURE_DISPLAY_NAMES.get(feature_names[i], feature_names[i]) for i in sorted_idx]
    sorted_values = feature_values[sorted_idx]

    # Create labels with feature values
    labels = [f"{name}<br>({val:.2f})" if abs(val) < 1000 else f"{name}<br>({val:.0f})"
              for name, val in zip(sorted_names, sorted_values)]

    # Colors based on positive/negative
    colors = [RED if v > 0 else GREEN for v in sorted_shap]

    fig = go.Figure(go.Waterfall(
        name="SHAP",
        orientation="v",
        measure=["relative"] * len(sorted_shap) + ["total"],
        x=labels + ["Prediction"],
        y=list(sorted_shap) + [0],
        connector={"line": {"color": "rgba(255,255,255,0.3)"}},
        decreasing={"marker": {"color": GREEN}},
        increasing={"marker": {"color": RED}},
        totals={"marker": {"color": BLUE}},
        base=base_value
    ))

    fig.update_layout(
        title="Feature Contributions to Prediction",
        template="plotly_dark",
        height=500,
        showlegend=False,
        yaxis_title="Log-odds contribution",
        xaxis_tickangle=-45
    )

    return fig


def create_force_plot(shap_values, feature_names, base_value, prediction):
    """Create a horizontal bar chart showing SHAP contributions."""
    # Sort by SHAP value (not absolute)
    sorted_idx = np.argsort(shap_values)

    sorted_shap = shap_values[sorted_idx]
    sorted_names = [FEATURE_DISPLAY_NAMES.get(feature_names[i], feature_names[i]) for i in sorted_idx]

    colors = [RED if v > 0 else GREEN for v in sorted_shap]

    fig = go.Figure(go.Bar(
        x=sorted_shap,
        y=sorted_names,
        orientation='h',
        marker_color=colors,
        text=[f"{v:+.3f}" for v in sorted_shap],
        textposition='outside'
    ))

    fig.add_vline(x=0, line_dash="dash", line_color="white", opacity=0.5)

    fig.update_layout(
        title=f"SHAP Force Plot (Base: {base_value:.3f}, Prediction: {prediction:.1%})",
        template="plotly_dark",
        height=400,
        xaxis_title="SHAP Value (impact on log-odds)",
        yaxis_title="Feature",
        showlegend=False
    )

    return fig


def create_global_importance_plot(shap_values, feature_names):
    """Create a bar chart of mean absolute SHAP values."""
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

    # Sort by importance
    sorted_idx = np.argsort(mean_abs_shap)[::-1]
    sorted_importance = mean_abs_shap[sorted_idx]
    sorted_names = [FEATURE_DISPLAY_NAMES.get(feature_names[i], feature_names[i]) for i in sorted_idx]

    fig = go.Figure(go.Bar(
        x=sorted_importance,
        y=sorted_names,
        orientation='h',
        marker_color=BLUE,
        text=[f"{v:.4f}" for v in sorted_importance],
        textposition='outside'
    ))

    fig.update_layout(
        title="Global Feature Importance (Mean |SHAP Value|)",
        template="plotly_dark",
        height=450,
        xaxis_title="Mean |SHAP Value|",
        yaxis_title="Feature",
        yaxis=dict(autorange="reversed")
    )

    return fig


def create_beeswarm_plotly(shap_values, X_scaled, feature_names):
    """Create a beeswarm-style plot using Plotly."""
    # Create a summary plot as scatter
    fig = go.Figure()

    mean_abs = np.mean(np.abs(shap_values), axis=0)
    sorted_idx = np.argsort(mean_abs)[::-1]

    for rank, i in enumerate(sorted_idx):
        feature_shap = shap_values[:, i]
        feature_val = X_scaled[:, i]

        # Normalize feature values for color
        vmin, vmax = feature_val.min(), feature_val.max()
        if vmax > vmin:
            colors = (feature_val - vmin) / (vmax - vmin)
        else:
            colors = np.zeros_like(feature_val)

        # Add jitter for y
        y_jitter = rank + np.random.uniform(-0.3, 0.3, len(feature_shap))

        fig.add_trace(go.Scatter(
            x=feature_shap,
            y=y_jitter,
            mode='markers',
            marker=dict(
                size=4,
                color=colors,
                colorscale='RdBu_r',
                opacity=0.6
            ),
            name=FEATURE_DISPLAY_NAMES.get(feature_names[i], feature_names[i]),
            hovertemplate=f"{feature_names[i]}<br>SHAP: %{{x:.3f}}<extra></extra>"
        ))

    sorted_names = [FEATURE_DISPLAY_NAMES.get(feature_names[i], feature_names[i]) for i in sorted_idx]

    fig.update_layout(
        title="SHAP Summary Plot (Feature Impact Distribution)",
        template="plotly_dark",
        height=500,
        xaxis_title="SHAP Value (impact on prediction)",
        yaxis=dict(
            tickmode='array',
            tickvals=list(range(len(sorted_names))),
            ticktext=sorted_names
        ),
        showlegend=False
    )

    fig.add_vline(x=0, line_dash="dash", line_color="white", opacity=0.3)

    return fig


def generate_explanation_text(shap_values, feature_names, feature_values):
    """Generate plain English explanation of top factors."""
    # Get top 3 positive and negative contributors
    sorted_idx = np.argsort(shap_values)

    positive_factors = []
    negative_factors = []

    for i in sorted_idx[::-1]:  # Start from highest (most risk-increasing)
        if shap_values[i] > 0.01 and len(positive_factors) < 3:
            name = FEATURE_DISPLAY_NAMES.get(feature_names[i], feature_names[i])
            positive_factors.append((name, shap_values[i], feature_values[i]))

    for i in sorted_idx:  # Start from lowest (most risk-decreasing)
        if shap_values[i] < -0.01 and len(negative_factors) < 3:
            name = FEATURE_DISPLAY_NAMES.get(feature_names[i], feature_names[i])
            negative_factors.append((name, shap_values[i], feature_values[i]))

    return positive_factors, negative_factors


# ========== PAGE CONTENT ==========
st.title("Model Explainability")
st.markdown("Understand what drives the model's predictions using SHAP values")

# Load model
model = load_model()
if model is None:
    st.error("Model not found. Please run `python models/train_model.py` first.")
    st.stop()

st.markdown("---")

# ========== SECTION 1: INDIVIDUAL PREDICTION EXPLANATION ==========
st.markdown("## 1. Individual Prediction Explanation")
st.markdown("Enter borrower details to see what factors drive the prediction")

with st.form("explain_form"):
    col1, col2 = st.columns(2)

    with col1:
        income = st.number_input(
            "Annual Income ($)",
            min_value=10000,
            max_value=10000000,
            value=200000,
            step=10000
        )

        loan_amount = st.number_input(
            "Loan Amount ($)",
            min_value=10000,
            max_value=5000000,
            value=500000,
            step=10000
        )

        annuity = st.number_input(
            "Annual Repayment ($)",
            min_value=1000,
            max_value=200000,
            value=25000,
            step=1000
        )

        age = st.slider("Age (years)", 18, 70, 35)

    with col2:
        employed_years = st.slider("Years Employed", 0, 40, 5)
        ext_source_1 = st.slider("Bureau Score 1", 0.0, 1.0, 0.5, 0.01)
        ext_source_2 = st.slider("Bureau Score 2", 0.0, 1.0, 0.5, 0.01)
        ext_source_3 = st.slider("Bureau Score 3", 0.0, 1.0, 0.5, 0.01)

    explain_btn = st.form_submit_button("Explain Prediction", type="primary", use_container_width=True)

if explain_btn:
    # Prepare input
    input_data = pd.DataFrame([{
        'AMT_INCOME_TOTAL': income,
        'AMT_CREDIT': loan_amount,
        'AMT_ANNUITY': annuity,
        'DAYS_BIRTH': age * -365,
        'DAYS_EMPLOYED': employed_years * -365,
        'EXT_SOURCE_1': ext_source_1,
        'EXT_SOURCE_2': ext_source_2,
        'EXT_SOURCE_3': ext_source_3,
        'DEBT_TO_INCOME': loan_amount / (income + 1),
        'ANNUITY_TO_INCOME': annuity / (income + 1)
    }])

    # Get prediction
    prob = model.predict_proba(input_data)[0][1]
    risk_label, risk_color = get_risk_label(prob)

    # Transform input through pipeline for SHAP
    imputer = model.named_steps['imputer']
    scaler = model.named_steps['scaler']
    classifier = model.named_steps['classifier']

    X_imputed = imputer.transform(input_data[FEATURE_NAMES])
    X_scaled = scaler.transform(X_imputed)

    # Get SHAP values
    explainer = shap.LinearExplainer(classifier, X_scaled)
    shap_values = explainer.shap_values(X_scaled)[0]
    base_value = explainer.expected_value

    # Display prediction
    st.markdown("### Prediction Result")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Default Probability", f"{prob*100:.1f}%")

    with col2:
        st.markdown(f"""
        <div style="padding: 1rem; border-radius: 0.5rem; background-color: {risk_color}20; border: 2px solid {risk_color}; text-align: center;">
            <h3 style="color: {risk_color}; margin: 0;">{risk_label}</h3>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.metric("SHAP Base Value", f"{base_value:.3f}")

    st.markdown("---")

    # SHAP Visualizations
    st.markdown("### SHAP Feature Contributions")

    tab1, tab2 = st.tabs(["Force Plot", "Waterfall Plot"])

    with tab1:
        force_fig = create_force_plot(shap_values, FEATURE_NAMES, base_value, prob)
        st.plotly_chart(force_fig, use_container_width=True)

    with tab2:
        waterfall_fig = create_waterfall_plot(
            shap_values,
            X_scaled[0],
            FEATURE_NAMES,
            base_value
        )
        st.plotly_chart(waterfall_fig, use_container_width=True)

    # Plain English Explanation
    st.markdown("### Plain English Explanation")

    positive_factors, negative_factors = generate_explanation_text(
        shap_values, FEATURE_NAMES, input_data[FEATURE_NAMES].values[0]
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        <div style="padding: 1rem; border-radius: 0.5rem; background-color: {RED}20; border: 1px solid {RED};">
            <h4 style="color: {RED}; margin: 0 0 0.5rem 0;">Factors Increasing Risk</h4>
        </div>
        """, unsafe_allow_html=True)

        if positive_factors:
            for name, shap_val, feat_val in positive_factors:
                if 'Income' in name:
                    val_str = f"${feat_val:,.0f}"
                elif 'Bureau' in name:
                    val_str = f"{feat_val:.2f}"
                elif 'days' in name.lower():
                    val_str = f"{abs(feat_val)/365:.1f} years"
                else:
                    val_str = f"{feat_val:.2f}"
                st.markdown(f"- **{name}** ({val_str}): +{shap_val:.3f}")
        else:
            st.markdown("_No significant risk-increasing factors_")

    with col2:
        st.markdown(f"""
        <div style="padding: 1rem; border-radius: 0.5rem; background-color: {GREEN}20; border: 1px solid {GREEN};">
            <h4 style="color: {GREEN}; margin: 0 0 0.5rem 0;">Factors Decreasing Risk</h4>
        </div>
        """, unsafe_allow_html=True)

        if negative_factors:
            for name, shap_val, feat_val in negative_factors:
                if 'Income' in name:
                    val_str = f"${feat_val:,.0f}"
                elif 'Bureau' in name:
                    val_str = f"{feat_val:.2f}"
                elif 'days' in name.lower():
                    val_str = f"{abs(feat_val)/365:.1f} years"
                else:
                    val_str = f"{feat_val:.2f}"
                st.markdown(f"- **{name}** ({val_str}): {shap_val:.3f}")
        else:
            st.markdown("_No significant risk-decreasing factors_")

    # Interpretation guide
    with st.expander("How to interpret SHAP values"):
        st.markdown("""
        **SHAP (SHapley Additive exPlanations)** values show how each feature contributes to the prediction:

        - **Positive SHAP value** → Feature pushes prediction **toward default** (higher risk)
        - **Negative SHAP value** → Feature pushes prediction **away from default** (lower risk)
        - **Magnitude** indicates the strength of the effect

        The model starts from a **base value** (average prediction) and each feature adds or subtracts from it.

        **Example:** If Bureau Score 2 has SHAP = -0.5, it means this borrower's bureau score
        significantly reduces their predicted default probability compared to average.
        """)

st.markdown("---")

# ========== SECTION 2: GLOBAL FEATURE IMPORTANCE ==========
st.markdown("## 2. Global Feature Importance")
st.markdown("Understanding which features matter most across all borrowers")

# Load sample data and compute global SHAP
sample_df = load_sample_data()

if sample_df is not None:
    with st.spinner("Computing SHAP values for 1,000 borrowers..."):
        shap_values_global, X_scaled_global = compute_global_shap_values(model, sample_df)

    # Global importance bar chart
    col1, col2 = st.columns(2)

    with col1:
        importance_fig = create_global_importance_plot(shap_values_global, FEATURE_NAMES)
        st.plotly_chart(importance_fig, use_container_width=True)

    with col2:
        st.markdown("### Key Insights")

        mean_abs_shap = np.mean(np.abs(shap_values_global), axis=0)
        sorted_idx = np.argsort(mean_abs_shap)[::-1]

        top_3_names = [FEATURE_DISPLAY_NAMES.get(FEATURE_NAMES[i], FEATURE_NAMES[i]) for i in sorted_idx[:3]]

        st.markdown(f"""
        The **top 3 most important features** for predicting default are:

        1. **{top_3_names[0]}** - Strongest predictor
        2. **{top_3_names[1]}** - Second most important
        3. **{top_3_names[2]}** - Third most important

        These features have the highest average impact on predictions across all borrowers.
        """)

        # Show feature importance table
        importance_df = pd.DataFrame({
            'Feature': [FEATURE_DISPLAY_NAMES.get(FEATURE_NAMES[i], FEATURE_NAMES[i]) for i in sorted_idx],
            'Mean |SHAP|': [mean_abs_shap[i] for i in sorted_idx],
            'Rank': range(1, len(FEATURE_NAMES) + 1)
        })
        st.dataframe(importance_df, use_container_width=True, hide_index=True)

    # SHAP Summary Plot (Beeswarm)
    st.markdown("### SHAP Summary Plot")
    st.markdown("Each dot represents a borrower. Color shows feature value (red=high, blue=low).")

    beeswarm_fig = create_beeswarm_plotly(shap_values_global, X_scaled_global, FEATURE_NAMES)
    st.plotly_chart(beeswarm_fig, use_container_width=True)

    st.markdown("""
    **Reading the summary plot:**
    - Features are sorted by importance (top = most important)
    - Dots to the **right** of zero increase default risk
    - Dots to the **left** decrease default risk
    - Color indicates the feature value: high value (red), low value (blue)

    **Example:** For external bureau scores, we see blue dots (low scores) push predictions right
    (higher risk), while red dots (high scores) push left (lower risk). This confirms that
    higher bureau scores reduce default probability.
    """)

else:
    st.warning("Could not load sample data. Ensure the database exists at `database/credit_risk.db`.")

# About section
st.markdown("---")
with st.expander("About SHAP"):
    st.markdown("""
    **SHAP (SHapley Additive exPlanations)** is a game-theoretic approach to explain machine learning predictions.

    **Key concepts:**
    - Based on Shapley values from cooperative game theory
    - Provides consistent and locally accurate explanations
    - Shows both direction and magnitude of feature effects

    **In this dashboard:**
    - We use `LinearExplainer` optimized for linear models like Logistic Regression
    - SHAP values are computed on scaled features
    - Global importance is averaged across 1,000 sample borrowers

    **Reference:** Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions.
    """)
