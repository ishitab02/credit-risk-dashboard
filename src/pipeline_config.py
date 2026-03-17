TABLE_CONFIGS = {
    "applications": {
        "source_file": "application_train.csv",
        "primary_key": "SK_ID_CURR",
        "expected_row_range": (300_000, 320_000),
        "required_columns": [
            "SK_ID_CURR", "TARGET", "AMT_INCOME_TOTAL", "AMT_CREDIT",
            "AMT_ANNUITY", "AMT_GOODS_PRICE", "DAYS_BIRTH", "DAYS_EMPLOYED",
            "NAME_CONTRACT_TYPE", "CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY",
            "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS",
            "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
        ],
        "column_types": {
            "SK_ID_CURR": "int64",
            "TARGET": "float64",
            "AMT_INCOME_TOTAL": "float64",
            "AMT_CREDIT": "float64",
            "AMT_ANNUITY": "float64",
            "DAYS_BIRTH": "float64",
            "DAYS_EMPLOYED": "float64",
        },
        "dq_checks": {
            "null_thresholds": {
                "SK_ID_CURR": 0.0,
                "TARGET": 0.0,
                "AMT_INCOME_TOTAL": 0.0,
                "AMT_CREDIT": 0.0,
                "EXT_SOURCE_1": 0.60,
                "EXT_SOURCE_2": 0.02,
                "EXT_SOURCE_3": 0.25,
            },
            "range_checks": {
                "TARGET": {"min": 0, "max": 1, "type": "inclusive"},
                "AMT_INCOME_TOTAL": {"min": 0, "max": None, "type": "lower_bound"},
                "AMT_CREDIT": {"min": 0, "max": None, "type": "lower_bound"},
                "DAYS_BIRTH": {"min": None, "max": 0, "type": "upper_bound"},
            },
            "unique_checks": ["SK_ID_CURR"],
            "sentinel_checks": {
                "DAYS_EMPLOYED": {"value": 365243, "max_pct": 0.20},
            },
        },
    },
    "bureau": {
        "source_file": "bureau.csv",
        "primary_key": "SK_ID_BUREAU",
        "foreign_key": {"column": "SK_ID_CURR", "references": "applications.SK_ID_CURR"},
        "expected_row_range": (1_700_000, 1_750_000),
        "required_columns": [
            "SK_ID_CURR", "SK_ID_BUREAU", "CREDIT_ACTIVE", "CREDIT_TYPE",
            "AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT", "AMT_CREDIT_SUM_OVERDUE",
            "CREDIT_DAY_OVERDUE",
        ],
        "column_types": {
            "SK_ID_CURR": "int64",
            "SK_ID_BUREAU": "int64",
            "AMT_CREDIT_SUM": "float64",
        },
        "dq_checks": {
            "null_thresholds": {
                "SK_ID_CURR": 0.0,
                "SK_ID_BUREAU": 0.0,
                "CREDIT_ACTIVE": 0.0,
            },
            "range_checks": {
                "CREDIT_DAY_OVERDUE": {"min": 0, "max": None, "type": "lower_bound"},
            },
            "unique_checks": ["SK_ID_BUREAU"],
            "sentinel_checks": {},
        },
    },
    "previous_application": {
        "source_file": "previous_application.csv",
        "primary_key": "SK_ID_PREV",
        "foreign_key": {"column": "SK_ID_CURR", "references": "applications.SK_ID_CURR"},
        "expected_row_range": (1_650_000, 1_700_000),
        "required_columns": [
            "SK_ID_CURR", "SK_ID_PREV", "NAME_CONTRACT_STATUS",
            "AMT_APPLICATION", "AMT_CREDIT", "NAME_CONTRACT_TYPE",
        ],
        "column_types": {
            "SK_ID_CURR": "int64",
            "SK_ID_PREV": "int64",
            "AMT_APPLICATION": "float64",
        },
        "dq_checks": {
            "null_thresholds": {
                "SK_ID_CURR": 0.0,
                "SK_ID_PREV": 0.0,
                "NAME_CONTRACT_STATUS": 0.0,
            },
            "range_checks": {},
            "unique_checks": ["SK_ID_PREV"],
            "sentinel_checks": {},
        },
    },
}

STATUS_SUCCESS = "success"
STATUS_FAILED = "failed"
STATUS_SKIPPED = "skipped"

READ_CHUNKSIZE = 10_000
WRITE_CHUNKSIZE = 1_000