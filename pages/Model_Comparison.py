import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os

st.set_page_config(
    page_title="Model Comparison | Credit Risk",
    page_icon="MC",
    layout="wide"
)

# Colors
LOGREG_COLOR = "#3498db"
XGBOOST_COLOR = "#9b59b6"
GREEN = "#2ecc71"
RED = "#e74c3c"
ORANGE = "#f39c12"


@st.cache_resource
def load_models():
    """Load both trained models."""
    logreg_path = "models/logreg_model.pkl"
    xgb_path = "models/xgboost_model.pkl"

    logreg = None
    xgb = None

    if os.path.exists(logreg_path):
        logreg = joblib.load(logreg_path)
    if os.path.exists(xgb_path):
        xgb = joblib.load(xgb_path)

    return logreg, xgb


@st.cache_data
def load_metrics():
    """Load pre-computed metrics from training."""
    metrics_path = "models/model_metrics.pkl"
    if os.path.exists(metrics_path):
        return joblib.load(metrics_path)
    return None


def get_risk_label(prob):
    """Returns risk label and color based on probability."""
    if prob < 0.15:
        return "Low Risk", GREEN
    elif prob < 0.30:
        return "Medium Risk", ORANGE
    else:
        return "High Risk", RED


# ========== PAGE HEADER ==========
st.title("Model Comparison")
st.markdown("Compare Logistic Regression and XGBoost model performance")

# Load models and metrics
logreg_model, xgb_model = load_models()
metrics = load_metrics()

if logreg_model is None or xgb_model is None or metrics is None:
    st.error("Models or metrics not found. Please run `python models/train_model.py` first.")
    st.stop()

st.markdown("---")

# ========== SECTION 1: PERFORMANCE METRICS TABLE ==========
st.markdown("### Performance Metrics Comparison")

col1, col2 = st.columns(2)

with col1:
    st.markdown(f"""
    <div style="padding: 1rem; border-radius: 0.5rem; background-color: {LOGREG_COLOR}20; border: 2px solid {LOGREG_COLOR}; text-align: center;">
        <h4 style="color: {LOGREG_COLOR}; margin: 0;">Logistic Regression</h4>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div style="padding: 1rem; border-radius: 0.5rem; background-color: {XGBOOST_COLOR}20; border: 2px solid {XGBOOST_COLOR}; text-align: center;">
        <h4 style="color: {XGBOOST_COLOR}; margin: 0;">XGBoost</h4>
    </div>
    """, unsafe_allow_html=True)

# Metrics table
metrics_df = pd.DataFrame({
    'Metric': ['ROC-AUC', 'Precision', 'Recall', 'F1 Score'],
    'Logistic Regression': [
        f"{metrics['logreg']['auc']:.4f}",
        f"{metrics['logreg']['precision']:.4f}",
        f"{metrics['logreg']['recall']:.4f}",
        f"{metrics['logreg']['f1']:.4f}"
    ],
    'XGBoost': [
        f"{metrics['xgboost']['auc']:.4f}",
        f"{metrics['xgboost']['precision']:.4f}",
        f"{metrics['xgboost']['recall']:.4f}",
        f"{metrics['xgboost']['f1']:.4f}"
    ]
})

st.dataframe(
    metrics_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        'Metric': st.column_config.TextColumn('Metric', width='medium'),
        'Logistic Regression': st.column_config.TextColumn('Logistic Regression', width='medium'),
        'XGBoost': st.column_config.TextColumn('XGBoost', width='medium')
    }
)

# Highlight winner for each metric
logreg_m = metrics['logreg']
xgb_m = metrics['xgboost']

winners = []
if logreg_m['auc'] > xgb_m['auc']:
    winners.append(("ROC-AUC", "Logistic Regression", logreg_m['auc'] - xgb_m['auc']))
else:
    winners.append(("ROC-AUC", "XGBoost", xgb_m['auc'] - logreg_m['auc']))

if logreg_m['f1'] > xgb_m['f1']:
    winners.append(("F1 Score", "Logistic Regression", logreg_m['f1'] - xgb_m['f1']))
else:
    winners.append(("F1 Score", "XGBoost", xgb_m['f1'] - logreg_m['f1']))

st.markdown("**Key Insights:**")
for metric, winner, diff in winners:
    st.markdown(f"- {winner} leads in {metric} by {diff:.4f}")

st.markdown("---")

# ========== SECTION 2: ROC CURVES ==========
st.markdown("### ROC Curve Comparison")

fig_roc = go.Figure()

# Logistic Regression ROC
fig_roc.add_trace(go.Scatter(
    x=metrics['logreg']['fpr'],
    y=metrics['logreg']['tpr'],
    mode='lines',
    name=f"LogReg (AUC={metrics['logreg']['auc']:.3f})",
    line=dict(color=LOGREG_COLOR, width=2)
))

# XGBoost ROC
fig_roc.add_trace(go.Scatter(
    x=metrics['xgboost']['fpr'],
    y=metrics['xgboost']['tpr'],
    mode='lines',
    name=f"XGBoost (AUC={metrics['xgboost']['auc']:.3f})",
    line=dict(color=XGBOOST_COLOR, width=2)
))

# Diagonal reference line
fig_roc.add_trace(go.Scatter(
    x=[0, 1],
    y=[0, 1],
    mode='lines',
    name='Random Classifier',
    line=dict(color='gray', width=1, dash='dash')
))

fig_roc.update_layout(
    title='ROC Curves - Model Comparison',
    xaxis_title='False Positive Rate',
    yaxis_title='True Positive Rate',
    template='plotly_dark',
    height=500,
    legend=dict(x=0.6, y=0.1),
    xaxis=dict(range=[0, 1]),
    yaxis=dict(range=[0, 1])
)

st.plotly_chart(fig_roc, use_container_width=True)

st.markdown("---")

# ========== SECTION 3: FEATURE IMPORTANCE COMPARISON ==========
st.markdown("### Feature Importance Comparison")

features = metrics['features']
logreg_importance = metrics['logreg']['coefficients']
xgb_importance = metrics['xgboost']['feature_importances']

# Normalize for comparison
logreg_norm = logreg_importance / logreg_importance.max()
xgb_norm = xgb_importance / xgb_importance.max()

# Create side-by-side bar charts
fig_importance = make_subplots(
    rows=1, cols=2,
    subplot_titles=['Logistic Regression (|Coefficients|)', 'XGBoost (Feature Importance)'],
    horizontal_spacing=0.15
)

# Sort by LogReg importance for consistent ordering
sort_idx = np.argsort(logreg_norm)[::-1]
sorted_features = [features[i] for i in sort_idx]
sorted_logreg = logreg_norm[sort_idx]
sorted_xgb = xgb_norm[sort_idx]

# Logistic Regression bars
fig_importance.add_trace(
    go.Bar(
        y=sorted_features,
        x=sorted_logreg,
        orientation='h',
        marker_color=LOGREG_COLOR,
        name='LogReg'
    ),
    row=1, col=1
)

# XGBoost bars (same order)
fig_importance.add_trace(
    go.Bar(
        y=sorted_features,
        x=sorted_xgb,
        orientation='h',
        marker_color=XGBOOST_COLOR,
        name='XGBoost'
    ),
    row=1, col=2
)

fig_importance.update_layout(
    template='plotly_dark',
    height=500,
    showlegend=False
)

fig_importance.update_xaxes(title_text='Normalized Importance', row=1, col=1)
fig_importance.update_xaxes(title_text='Normalized Importance', row=1, col=2)

st.plotly_chart(fig_importance, use_container_width=True)

# Feature importance insights
st.markdown("**Key Observations:**")
top_logreg = sorted_features[:3]
xgb_sort_idx = np.argsort(xgb_norm)[::-1]
top_xgb = [features[i] for i in xgb_sort_idx[:3]]

st.markdown(f"- **LogReg** top features: {', '.join(top_logreg)}")
st.markdown(f"- **XGBoost** top features: {', '.join(top_xgb)}")

if top_logreg != top_xgb:
    st.markdown("- Models weigh features differently, which may explain prediction disagreements")

st.markdown("---")

# ========== SECTION 4: PREDICTION COMPARISON TOOL ==========
st.markdown("### Prediction Comparison Tool")
st.markdown("Enter borrower details to see predictions from both models")

with st.form("prediction_form"):
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

    predict_btn = st.form_submit_button("Compare Predictions", type="primary", use_container_width=True)

if predict_btn:
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

    # Get predictions
    prob_logreg = logreg_model.predict_proba(input_data)[0][1]
    prob_xgb = xgb_model.predict_proba(input_data)[0][1]

    logreg_label, logreg_color = get_risk_label(prob_logreg)
    xgb_label, xgb_color = get_risk_label(prob_xgb)

    st.markdown("#### Prediction Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div style="padding: 1.5rem; border-radius: 0.5rem; background-color: {LOGREG_COLOR}20; border: 2px solid {LOGREG_COLOR}; text-align: center;">
            <h4 style="color: {LOGREG_COLOR}; margin: 0;">Logistic Regression</h4>
            <h2 style="color: white; margin: 0.5rem 0;">{prob_logreg*100:.1f}%</h2>
            <p style="color: {logreg_color}; margin: 0; font-weight: bold;">{logreg_label}</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div style="padding: 1.5rem; border-radius: 0.5rem; background-color: {XGBOOST_COLOR}20; border: 2px solid {XGBOOST_COLOR}; text-align: center;">
            <h4 style="color: {XGBOOST_COLOR}; margin: 0;">XGBoost</h4>
            <h2 style="color: white; margin: 0.5rem 0;">{prob_xgb*100:.1f}%</h2>
            <p style="color: {xgb_color}; margin: 0; font-weight: bold;">{xgb_label}</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        diff = abs(prob_logreg - prob_xgb) * 100
        diff_color = RED if diff > 10 else ORANGE if diff > 5 else GREEN

        st.markdown(f"""
        <div style="padding: 1.5rem; border-radius: 0.5rem; background-color: {diff_color}20; border: 2px solid {diff_color}; text-align: center;">
            <h4 style="color: white; margin: 0;">Difference</h4>
            <h2 style="color: {diff_color}; margin: 0.5rem 0;">{diff:.1f}%</h2>
            <p style="color: {diff_color}; margin: 0; font-weight: bold;">{"Significant" if diff > 10 else "Moderate" if diff > 5 else "Aligned"}</p>
        </div>
        """, unsafe_allow_html=True)

    # Agreement indicator
    if diff > 10:
        st.warning(f"Models disagree significantly ({diff:.1f}% difference). Consider reviewing the input or consulting both predictions.")
    elif diff > 5:
        st.info(f"Models show moderate difference ({diff:.1f}%). Predictions are reasonably aligned.")
    else:
        st.success(f"Models agree closely ({diff:.1f}% difference). High confidence in prediction.")

st.markdown("---")

# ========== SECTION 5: MODEL SELECTION FOR SCORER ==========
st.markdown("### Model Selection for Default Scorer")
st.markdown("Select which model to use on the Default Scorer page")

# Initialize session state if not exists
if 'selected_model' not in st.session_state:
    st.session_state['selected_model'] = 'logreg'

selected = st.radio(
    "Select model for scoring:",
    options=['logreg', 'xgboost'],
    format_func=lambda x: 'Logistic Regression' if x == 'logreg' else 'XGBoost',
    index=0 if st.session_state['selected_model'] == 'logreg' else 1,
    horizontal=True
)

if selected != st.session_state['selected_model']:
    st.session_state['selected_model'] = selected
    st.success(f"Model updated to {'Logistic Regression' if selected == 'logreg' else 'XGBoost'}. Go to Default Scorer to use it.")

# Show current selection
model_name = 'Logistic Regression' if st.session_state['selected_model'] == 'logreg' else 'XGBoost'
model_color = LOGREG_COLOR if st.session_state['selected_model'] == 'logreg' else XGBOOST_COLOR

st.markdown(f"""
<div style="padding: 1rem; border-radius: 0.5rem; background-color: {model_color}20; border: 2px solid {model_color};">
    <p style="color: white; margin: 0;"><strong>Current Selection:</strong> {model_name}</p>
    <p style="color: gray; margin: 0; font-size: 0.9em;">This model will be used on the Default Scorer page</p>
</div>
""", unsafe_allow_html=True)

# Model recommendations
st.markdown("---")
st.markdown("#### Model Recommendations")

col1, col2 = st.columns(2)

with col1:
    st.markdown(f"""
    **Logistic Regression**
    - More interpretable
    - Faster inference
    - Better for regulatory compliance
    - May miss non-linear patterns
    """)

with col2:
    st.markdown(f"""
    **XGBoost**
    - Often higher accuracy
    - Captures complex patterns
    - Handles feature interactions
    - Less interpretable (black box)
    """)
