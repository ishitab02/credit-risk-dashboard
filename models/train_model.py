import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, precision_score, recall_score, f1_score, roc_curve
from xgboost import XGBClassifier
import joblib

from src.transforms import add_derived_features


FEATURES = [
    'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY',
    'DAYS_BIRTH', 'DAYS_EMPLOYED',
    'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3',
    'DEBT_TO_INCOME', 'ANNUITY_TO_INCOME'
]


def train_models():
    """Trains and saves both Logistic Regression and XGBoost models."""

    print("=" * 60)
    print("Credit Risk Model Training - Dual Model Comparison")
    print("=" * 60)

    # Check if database exists
    db_path = "database/credit_risk.db"
    if not os.path.exists(db_path):
        print(f"Error: Database not found at {db_path}")
        print("Please run src/db_setup.py first")
        return

    # Load data from SQLite
    print("\n1. Loading data from database...")
    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False}
    )

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
    df = pd.read_sql(sql, engine)
    print(f"   Loaded {len(df):,} rows")

    print("\n2. Engineering features...")
    df = add_derived_features(df)

    df.loc[df['DAYS_EMPLOYED'] > 0, 'DAYS_EMPLOYED'] = 0

    X = df[FEATURES]
    y = df['TARGET']

    print(f"   Features: {FEATURES}")
    print(f"   Target distribution: {y.value_counts().to_dict()}")
    default_rate = y.mean()
    print(f"   Default rate: {default_rate*100:.2f}%")

    # Calculate scale_pos_weight for XGBoost 
    scale_pos_weight = (1 - default_rate) / default_rate
    print(f"   Scale pos weight for XGBoost: {scale_pos_weight:.2f}")

    # Train/test split 
    print("\n3. Splitting data (80/20, stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    print(f"   Training set: {len(X_train):,} samples")
    print(f"   Test set: {len(X_test):,} samples")

    # Preprocessing pipeline 
    preprocessor_logreg = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    preprocessor_xgb = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # ========== LOGISTIC REGRESSION ==========
    print("\n4. Training Logistic Regression model...")
    logreg_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(
            max_iter=1000,
            C=0.1,
            class_weight='balanced',
            random_state=42
        ))
    ])

    logreg_pipeline.fit(X_train, y_train)
    print("   Logistic Regression trained successfully")

    # Evaluate LogReg
    y_pred_logreg = logreg_pipeline.predict(X_test)
    y_prob_logreg = logreg_pipeline.predict_proba(X_test)[:, 1]

    logreg_auc = roc_auc_score(y_test, y_prob_logreg)
    logreg_precision = precision_score(y_test, y_pred_logreg)
    logreg_recall = recall_score(y_test, y_pred_logreg)
    logreg_f1 = f1_score(y_test, y_pred_logreg)

    print(f"\n   Logistic Regression Results:")
    print(f"   ROC-AUC: {logreg_auc:.4f}")
    print(f"   Precision: {logreg_precision:.4f}")
    print(f"   Recall: {logreg_recall:.4f}")
    print(f"   F1 Score: {logreg_f1:.4f}")

    # ========== XGBOOST ==========
    print("\n5. Training XGBoost model...")
    xgb_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('classifier', XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric='auc'
        ))
    ])

    xgb_pipeline.fit(X_train, y_train)
    print("   XGBoost trained successfully")

    # Evaluate XGBoost
    y_pred_xgb = xgb_pipeline.predict(X_test)
    y_prob_xgb = xgb_pipeline.predict_proba(X_test)[:, 1]

    xgb_auc = roc_auc_score(y_test, y_prob_xgb)
    xgb_precision = precision_score(y_test, y_pred_xgb)
    xgb_recall = recall_score(y_test, y_pred_xgb)
    xgb_f1 = f1_score(y_test, y_pred_xgb)

    print(f"\n   XGBoost Results:")
    print(f"   ROC-AUC: {xgb_auc:.4f}")
    print(f"   Precision: {xgb_precision:.4f}")
    print(f"   Recall: {xgb_recall:.4f}")
    print(f"   F1 Score: {xgb_f1:.4f}")

    # ========== COMPARISON SUMMARY ==========
    print("\n" + "=" * 60)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Metric':<15} {'LogReg':>12} {'XGBoost':>12}")
    print("-" * 40)
    print(f"{'ROC-AUC':<15} {logreg_auc:>12.4f} {xgb_auc:>12.4f}")
    print(f"{'Precision':<15} {logreg_precision:>12.4f} {xgb_precision:>12.4f}")
    print(f"{'Recall':<15} {logreg_recall:>12.4f} {xgb_recall:>12.4f}")
    print(f"{'F1 Score':<15} {logreg_f1:>12.4f} {xgb_f1:>12.4f}")

    # Save models
    print("\n6. Saving models...")
    os.makedirs("models", exist_ok=True)

    logreg_path = "models/logreg_model.pkl"
    joblib.dump(logreg_pipeline, logreg_path)
    print(f"   Logistic Regression saved to {logreg_path}")

    xgb_path = "models/xgboost_model.pkl"
    joblib.dump(xgb_pipeline, xgb_path)
    print(f"   XGBoost saved to {xgb_path}")

    default_path = "models/default_scorer.pkl"
    joblib.dump(logreg_pipeline, default_path)
    print(f"   Default scorer (LogReg) saved to {default_path}")

    # ========== SAVE METRICS FOR COMPARISON PAGE ==========
    print("\n7. Saving evaluation metrics and data...")

    # Compute ROC curves
    fpr_logreg, tpr_logreg, _ = roc_curve(y_test, y_prob_logreg)
    fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_prob_xgb)

    # Get feature importances
    logreg_coefs = np.abs(logreg_pipeline.named_steps['classifier'].coef_[0])
    xgb_importances = xgb_pipeline.named_steps['classifier'].feature_importances_

    # Save metrics dictionary
    metrics = {
        'logreg': {
            'auc': logreg_auc,
            'precision': logreg_precision,
            'recall': logreg_recall,
            'f1': logreg_f1,
            'fpr': fpr_logreg,
            'tpr': tpr_logreg,
            'coefficients': logreg_coefs
        },
        'xgboost': {
            'auc': xgb_auc,
            'precision': xgb_precision,
            'recall': xgb_recall,
            'f1': xgb_f1,
            'fpr': fpr_xgb,
            'tpr': tpr_xgb,
            'feature_importances': xgb_importances
        },
        'features': FEATURES
    }

    metrics_path = "models/model_metrics.pkl"
    joblib.dump(metrics, metrics_path)
    print(f"   Metrics saved to {metrics_path}")

    print("\n8. Verification test...")
    test_input = pd.DataFrame([{
        'AMT_INCOME_TOTAL': 200000,
        'AMT_CREDIT': 500000,
        'AMT_ANNUITY': 25000,
        'DAYS_BIRTH': -35 * 365,
        'DAYS_EMPLOYED': -5 * 365,
        'EXT_SOURCE_1': 0.5,
        'EXT_SOURCE_2': 0.5,
        'EXT_SOURCE_3': 0.5,
        'DEBT_TO_INCOME': 500000 / 200001,
        'ANNUITY_TO_INCOME': 25000 / 200001
    }])

    prob_logreg = logreg_pipeline.predict_proba(test_input)[0][1]
    prob_xgb = xgb_pipeline.predict_proba(test_input)[0][1]
    print(f"   Test borrower - LogReg: {prob_logreg*100:.1f}%, XGBoost: {prob_xgb*100:.1f}%")

    print("\n" + "=" * 60)
    print("Training complete! Both models saved.")
    print("=" * 60)


if __name__ == "__main__":
    train_models()
