import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(
    page_title="Default Scorer | Credit Risk",
    page_icon="DS",
    layout="wide"
)


@st.cache_resource
def load_model(model_type):
    """Load the trained model based on selection."""
    if model_type == 'xgboost':
        model_path = "models/xgboost_model.pkl"
    else:
        model_path = "models/logreg_model.pkl"

    # Fallback to default_scorer.pkl if specific model not found
    if not os.path.exists(model_path):
        model_path = "models/default_scorer.pkl"

    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None


def get_risk_label(prob):
    """Returns risk label and color based on probability."""
    if prob < 0.15:
        return "Low Risk", "#2ecc71"
    elif prob < 0.30:
        return "Medium Risk", "#f39c12"
    else:
        return "High Risk", "#e74c3c"


st.title("Default Probability Scorer")
st.markdown("Predict the likelihood of loan default for a borrower")

# Get selected model from session state (default to logreg)
selected_model = st.session_state.get('selected_model', 'logreg')
model_name = 'Logistic Regression' if selected_model == 'logreg' else 'XGBoost'
model_color = '#3498db' if selected_model == 'logreg' else '#9b59b6'

# Show which model is active
st.markdown(f"""
<div style="padding: 0.5rem 1rem; border-radius: 0.5rem; background-color: {model_color}20; border: 1px solid {model_color}; margin-bottom: 1rem;">
    <span style="color: {model_color};">Active Model:</span> <strong>{model_name}</strong>
    <span style="color: gray; font-size: 0.85em;"> (Change in Model Comparison page)</span>
</div>
""", unsafe_allow_html=True)

# Load model
model = load_model(selected_model)
if model is None:
    st.error("Model not found. Please run `python models/train_model.py` first.")
    st.stop()

st.markdown("---")

# Input form
st.markdown("### Borrower Information")

with st.form("borrower_form"):
    col1, col2 = st.columns(2)

    with col1:
        income = st.number_input(
            "Annual Income ($)",
            min_value=10000,
            max_value=10000000,
            value=200000,
            step=10000,
            help="Total annual income"
        )

        loan_amount = st.number_input(
            "Loan Amount Requested ($)",
            min_value=10000,
            max_value=5000000,
            value=500000,
            step=10000,
            help="Total credit amount requested"
        )

        annuity = st.number_input(
            "Annual Repayment ($)",
            min_value=1000,
            max_value=200000,
            value=25000,
            step=1000,
            help="Loan annuity (annual payment)"
        )

        age = st.slider(
            "Age (years)",
            min_value=18,
            max_value=70,
            value=35,
            help="Borrower's age"
        )

    with col2:
        employed_years = st.slider(
            "Years Employed",
            min_value=0,
            max_value=40,
            value=5,
            help="Duration at current employment"
        )

        ext_source_1 = st.slider(
            "External Bureau Score 1",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.01,
            help="External credit score (0-1, higher is better)"
        )

        ext_source_2 = st.slider(
            "External Bureau Score 2",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.01,
            help="External credit score (0-1, higher is better)"
        )

        ext_source_3 = st.slider(
            "External Bureau Score 3",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.01,
            help="External credit score (0-1, higher is better)"
        )

    submitted = st.form_submit_button("Calculate Default Probability", type="primary", use_container_width=True)

if submitted:
    # Prepare input features
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

    st.markdown("---")
    st.markdown("### Prediction Results")

    # Display results
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="Default Probability",
            value=f"{prob * 100:.1f}%"
        )

    with col2:
        st.markdown(f"""
        <div style="padding: 1rem; border-radius: 0.5rem; background-color: {risk_color}20; border: 2px solid {risk_color};">
            <h3 style="color: {risk_color}; margin: 0; text-align: center;">{risk_label}</h3>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        avg_ext = (ext_source_1 + ext_source_2 + ext_source_3) / 3
        st.metric(
            label="Avg Bureau Score",
            value=f"{avg_ext:.2f}"
        )

    # Progress bar visualization
    st.markdown("### Risk Gauge")
    st.progress(float(min(prob, 1.0)))

    # Risk factors analysis
    st.markdown("---")
    st.markdown("### Risk Factor Analysis")

    factors = []

    # Bureau scores
    avg_bureau = (ext_source_1 + ext_source_2 + ext_source_3) / 3
    if avg_bureau < 0.3:
        factors.append(("[-]", "Low external bureau scores indicate higher risk"))
    elif avg_bureau > 0.6:
        factors.append(("+", "Strong external bureau scores reduce risk"))

    # Debt-to-income
    dti = loan_amount / income
    if dti > 5:
        factors.append(("[-]", f"High debt-to-income ratio ({dti:.1f}x) increases risk"))
    elif dti < 2:
        factors.append(("+", f"Low debt-to-income ratio ({dti:.1f}x) is favorable"))

    # Age
    if age < 25:
        factors.append(("[~]", "Younger borrowers historically show higher default rates"))
    elif age > 45:
        factors.append(("+", "Mature borrowers show lower default rates"))

    # Employment
    if employed_years < 1:
        factors.append(("[-]", "Short employment history increases risk"))
    elif employed_years > 5:
        factors.append(("+", "Stable employment history is favorable"))

    # Annuity ratio
    annuity_ratio = annuity / income
    if annuity_ratio > 0.3:
        factors.append(("[-]", f"High annuity-to-income ratio ({annuity_ratio:.1%}) strains repayment capacity"))

    if factors:
        for icon, text in factors:
            st.markdown(f"{icon} {text}")
    else:
        st.info("Borrower profile shows balanced risk factors")

    # Input summary
    with st.expander("Input Summary"):
        input_summary = pd.DataFrame({
            'Parameter': ['Annual Income', 'Loan Amount', 'Annual Payment', 'Age', 'Years Employed',
                         'Bureau Score 1', 'Bureau Score 2', 'Bureau Score 3', 'Debt-to-Income', 'Annuity-to-Income'],
            'Value': [f"${income:,.0f}", f"${loan_amount:,.0f}", f"${annuity:,.0f}", f"{age} years",
                     f"{employed_years} years", f"{ext_source_1:.2f}", f"{ext_source_2:.2f}",
                     f"{ext_source_3:.2f}", f"{dti:.2f}x", f"{annuity_ratio:.1%}"]
        })
        st.dataframe(input_summary, use_container_width=True, hide_index=True)

# Model info
st.markdown("---")
with st.expander("About the Model"):
    if selected_model == 'logreg':
        st.markdown("""
        **Model Type:** Logistic Regression with balanced class weights

        **Features Used:**
        - Annual Income, Loan Amount, Annual Repayment
        - Age (from DAYS_BIRTH)
        - Employment Duration (from DAYS_EMPLOYED)
        - External Bureau Scores (EXT_SOURCE_1, 2, 3)
        - Derived: Debt-to-Income, Annuity-to-Income ratios

        **Pipeline:**
        1. Median imputation for missing values
        2. StandardScaler normalization
        3. Logistic Regression (C=0.1, balanced weights)

        **Performance:** ROC-AUC ~0.72 on test set
        """)
    else:
        st.markdown("""
        **Model Type:** XGBoost Classifier

        **Features Used:**
        - Annual Income, Loan Amount, Annual Repayment
        - Age (from DAYS_BIRTH)
        - Employment Duration (from DAYS_EMPLOYED)
        - External Bureau Scores (EXT_SOURCE_1, 2, 3)
        - Derived: Debt-to-Income, Annuity-to-Income ratios

        **Pipeline:**
        1. Median imputation for missing values
        2. StandardScaler normalization
        3. XGBoost (100 trees, max_depth=5, learning_rate=0.1)

        **Performance:** ROC-AUC ~0.75 on test set (typically higher than LogReg)
        """)
