import pandas as pd

class ValidationResult:
    """Container for a single validation check result."""

    def __init__(self, check_name: str, passed: bool, details: str = ""):
        self.check_name = check_name
        self.passed = passed
        self.details = details

    def to_dict(self):
        return {
            "check_name": self.check_name,
            "passed": bool(self.passed),
            "details": str(self.details),
        }


def validate_schema(df: pd.DataFrame, table_config: dict) -> list[ValidationResult]:
    """
    Pre-load schema validation on a DataFrame chunk.
    Checks: required columns exist, column dtypes are coercible.
    """
    results = []

    required = set(table_config.get("required_columns", []))
    actual = set(df.columns)
    missing = required - actual
    results.append(ValidationResult(
        "required_columns",
        len(missing) == 0,
        f"Missing: {sorted(missing)}" if missing else f"All {len(required)} required columns present",
    ))

    for col, expected_dtype in table_config.get("column_types", {}).items():
        if col not in df.columns:
            continue
        try:
            df[col].astype(expected_dtype)
            results.append(ValidationResult(
                f"dtype_{col}",
                True,
                f"{col} coercible to {expected_dtype}",
            ))
        except (ValueError, TypeError):
            results.append(ValidationResult(
                f"dtype_{col}",
                False,
                f"{col} cannot be cast to {expected_dtype}",
            ))

    return results


def run_dq_checks(df: pd.DataFrame, table_config: dict) -> list[ValidationResult]:
    """
    Post-load data quality checks on a full table DataFrame.
    Checks: null rates, range constraints, uniqueness, sentinels.
    """
    results = []
    dq = table_config.get("dq_checks", {})
    total_rows = len(df)

    if total_rows == 0:
        results.append(ValidationResult("row_count", False, "Table is empty"))
        return results

    expected_range = table_config.get("expected_row_range")
    if expected_range:
        lo, hi = expected_range
        in_range = lo <= total_rows <= hi
        results.append(ValidationResult(
            "row_count_range",
            in_range,
            f"{total_rows:,} rows (expected {lo:,}-{hi:,})",
        ))

    for col, max_null_pct in dq.get("null_thresholds", {}).items():
        if col not in df.columns:
            continue
        null_rate = df[col].isnull().mean()
        passed = null_rate <= max_null_pct
        results.append(ValidationResult(
            f"null_rate_{col}",
            passed,
            f"{col}: {null_rate:.2%} null (threshold: {max_null_pct:.0%})",
        ))

    for col, spec in dq.get("range_checks", {}).items():
        if col not in df.columns:
            continue
        non_null = df[col].dropna()
        if len(non_null) == 0:
            continue

        violations = 0
        check_type = spec.get("type", "inclusive")

        if check_type == "inclusive":
            violations = ((non_null < spec["min"]) | (non_null > spec["max"])).sum()
        elif check_type == "lower_bound":
            violations = (non_null < spec["min"]).sum()
        elif check_type == "upper_bound":
            violations = (non_null > spec["max"]).sum()

        passed = violations == 0
        results.append(ValidationResult(
            f"range_{col}",
            passed,
            f"{col}: {violations:,} violations out of {len(non_null):,} values",
        ))

    for col in dq.get("unique_checks", []):
        if col not in df.columns:
            continue
        n_dupes = df[col].duplicated().sum()
        results.append(ValidationResult(
            f"unique_{col}",
            n_dupes == 0,
            f"{col}: {n_dupes:,} duplicate values",
        ))

    for col, spec in dq.get("sentinel_checks", {}).items():
        if col not in df.columns:
            continue
        sentinel_val = spec["value"]
        max_pct = spec["max_pct"]
        sentinel_rate = (df[col] == sentinel_val).mean()
        results.append(ValidationResult(
            f"sentinel_{col}",
            sentinel_rate <= max_pct,
            f"{col}: {sentinel_rate:.2%} sentinel ({sentinel_val}) (threshold: {max_pct:.0%})",
        ))

    return results


def check_referential_integrity(engine, fk_config: dict) -> ValidationResult:
    """
    Check that all FK values in child table exist in parent table.
    fk_config: {"column": "SK_ID_CURR", "references": "applications.SK_ID_CURR"}
    """
    fk_col = fk_config["column"]
    parent_table, parent_col = fk_config["references"].split(".")

    sql = f"""
    SELECT COUNT(*) as orphan_count
    FROM (
        SELECT DISTINCT {fk_col} FROM bureau
        EXCEPT
        SELECT DISTINCT {parent_col} FROM {parent_table}
    )
    """
    try:
        result = pd.read_sql(sql, engine)
        orphans = result["orphan_count"].iloc[0]
        return ValidationResult(
            f"ref_integrity_{fk_col}",
            orphans == 0,
            f"{orphans:,} orphan keys in {fk_col}",
        )
    except Exception as e:
        return ValidationResult(
            f"ref_integrity_{fk_col}",
            False,
            f"Check failed: {e}",
        )