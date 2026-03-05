# Credit Risk Analytics Dashboard

An interactive credit risk analytics dashboard built with Streamlit, analyzing 300,000+ loan applications to predict borrower default probability.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red)
![License](https://img.shields.io/badge/License-MIT-green)

## Live Demo

[View Live Dashboard](https://your-app-name.streamlit.app) *(Update after deployment)*

## Features

### 7 Interactive Pages

| Page | Description |
|------|-------------|
| **Home** | Dashboard overview with key metrics |
| **Portfolio Overview** | KPIs, income distributions, default rates by segment |
| **Risk Segmentation** | K-Means clustering into Low/Medium/High risk tiers |
| **Default Drivers** | Correlation analysis, feature importance heatmaps |
| **SQL Explorer** | Live SQL interface to query the database |
| **Default Scorer** | ML-powered default probability prediction |
| **Explainability** | SHAP waterfall plots explaining predictions |
| **Model Comparison** | LogReg vs XGBoost performance comparison |

### Technical Highlights

- **Data Pipeline**: SQLite database with 3 relational tables (applications, bureau, previous_application)
- **ML Models**: Logistic Regression (AUC: 0.72) and XGBoost (AUC: 0.75)
- **Explainability**: SHAP values for individual prediction explanations
- **Interactive SQL**: Safe query interface with dangerous command blocking

## Quick Start

### Local Development

```bash
# Clone the repository
git clone https://github.com/yourusername/credit-risk-dashboard.git
cd credit-risk-dashboard

# Install dependencies
pip install -r requirements.txt

# Run the app (auto-initializes DB and models on first run)
streamlit run app.py
```

### Using Full Dataset (Optional)

Download the [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk/data) dataset and place CSVs in `data/raw/`:
- `application_train.csv`
- `bureau.csv`
- `previous_application.csv`

Then delete `database/credit_risk.db` and restart the app.

## Project Structure

```
credit-risk-dashboard/
├── app.py                      # Main Streamlit entry point
├── requirements.txt            # Python dependencies
├── data/
│   └── sample/                 # Sampled data for deployment (50k rows)
├── database/                   # SQLite database (auto-generated)
├── models/
│   └── train_model.py          # Dual model training script
├── src/
│   ├── db_setup.py             # Database initialization
│   ├── queries.py              # SQL query functions
│   └── transforms.py           # Feature engineering
└── pages/
    ├── Portfolio_Overview.py
    ├── Risk_Segmentation.py
    ├── Default_Drivers.py
    ├── SQL_Explorer.py
    ├── Default_Scorer.py
    ├── Explainability.py
    └── Model_Comparison.py
```

## Tech Stack

- **Frontend**: Streamlit, Plotly
- **Database**: SQLite + SQLAlchemy
- **ML**: Scikit-learn, XGBoost, SHAP
- **Data**: Pandas, NumPy

## Model Performance

| Model | ROC-AUC | Precision | Recall | F1 |
|-------|---------|-----------|--------|-----|
| Logistic Regression | 0.722 | 0.15 | 0.65 | 0.25 |
| XGBoost | 0.753 | 0.17 | 0.68 | 0.27 |

*Note: Class imbalance (8% default rate) affects precision metrics. AUC is the primary evaluation metric.*

## Data Source

[Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) - Kaggle Competition Dataset

---

## License

MIT License - See [LICENSE](LICENSE) file
