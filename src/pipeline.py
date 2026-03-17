import hashlib
import json
import os
import time
import uuid
from datetime import datetime, timezone

import pandas as pd
from sqlalchemy import create_engine, text

from src.pipeline_config import (
    TABLE_CONFIGS, STATUS_SUCCESS, STATUS_FAILED, STATUS_SKIPPED,
    READ_CHUNKSIZE, WRITE_CHUNKSIZE,
)
from src.validators import validate_schema, run_dq_checks, check_referential_integrity


def _get_engine(db_path="database/credit_risk.db"):
    return create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
    )


def _init_pipeline_runs_table(engine):
    """Create pipeline_runs table if it doesn't exist."""
    create_sql = """
    CREATE TABLE IF NOT EXISTS pipeline_runs (
        run_id TEXT PRIMARY KEY,
        run_timestamp TEXT NOT NULL,
        table_name TEXT NOT NULL,
        status TEXT NOT NULL,
        rows_loaded INTEGER DEFAULT 0,
        rows_failed INTEGER DEFAULT 0,
        dq_checks_passed INTEGER DEFAULT 0,
        dq_checks_failed INTEGER DEFAULT 0,
        dq_details TEXT,
        file_hash TEXT,
        duration_secs REAL,
        error_message TEXT
    )
    """
    with engine.connect() as conn:
        conn.execute(text(create_sql))
        conn.commit()


def _compute_file_hash(filepath: str) -> str:
    """Compute MD5 hash of a file for change detection."""
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _get_last_hash(engine, table_name: str) -> str | None:
    """Get the file hash from the most recent successful run for a table."""
    sql = """
    SELECT file_hash FROM pipeline_runs
    WHERE table_name = :table AND status = 'success'
    ORDER BY run_timestamp DESC LIMIT 1
    """
    try:
        result = pd.read_sql(text(sql), engine, params={"table": table_name})
        if len(result) > 0:
            return result["file_hash"].iloc[0]
    except Exception:
        pass
    return None


def _log_run(engine, run_id, table_name, status, rows_loaded=0, rows_failed=0,
             dq_passed=0, dq_failed=0, dq_details=None, file_hash=None,
             duration=0.0, error_msg=None):
    """Insert a pipeline run record."""
    insert_sql = """
    INSERT INTO pipeline_runs
        (run_id, run_timestamp, table_name, status, rows_loaded, rows_failed,
         dq_checks_passed, dq_checks_failed, dq_details, file_hash,
         duration_secs, error_message)
    VALUES
        (:run_id, :ts, :table, :status, :rows_loaded, :rows_failed,
         :dq_passed, :dq_failed, :dq_details, :file_hash,
         :duration, :error_msg)
    """
    with engine.connect() as conn:
        conn.execute(text(insert_sql), {
            "run_id": run_id,
            "ts": datetime.now(timezone.utc).isoformat(),
            "table": table_name,
            "status": status,
            "rows_loaded": rows_loaded,
            "rows_failed": rows_failed,
            "dq_passed": dq_passed,
            "dq_failed": dq_failed,
            "dq_details": json.dumps(dq_details) if dq_details else None,
            "file_hash": file_hash,
            "duration": round(duration, 2),
            "error_msg": error_msg,
        })
        conn.commit()


def _resolve_data_dir() -> str:
    """Auto-detect data source directory."""
    if os.path.exists("data/sample/application_train.csv"):
        return "data/sample"
    elif os.path.exists("data/raw/application_train.csv"):
        return "data/raw"
    raise FileNotFoundError("No data found in data/sample/ or data/raw/")


def ingest_table(engine, table_name: str, table_config: dict, data_dir: str,
                 force: bool = False) -> dict:
    """
    Ingest a single table through the full ETL pipeline:
    1. Compute file hash for incremental detection
    2. Schema validation on first chunk
    3. Chunked load into SQLite
    4. Post-load DQ checks
    5. Log the run

    Returns a summary dict.
    """
    run_id = str(uuid.uuid4())[:8]
    start = time.time()
    source_file = os.path.join(data_dir, table_config["source_file"])

    if not os.path.exists(source_file):
        _log_run(engine, run_id, table_name, STATUS_FAILED,
                 error_msg=f"Source file not found: {source_file}")
        return {"table": table_name, "status": STATUS_FAILED, "error": "File not found"}

    file_hash = _compute_file_hash(source_file)
    last_hash = _get_last_hash(engine, table_name)

    if not force and last_hash == file_hash:
        _log_run(engine, run_id, table_name, STATUS_SKIPPED,
                 file_hash=file_hash, duration=time.time() - start)
        return {"table": table_name, "status": STATUS_SKIPPED, "reason": "No file changes detected"}

    first_chunk = pd.read_csv(source_file, nrows=1000)
    schema_results = validate_schema(first_chunk, table_config)
    schema_failures = [r for r in schema_results if not r.passed]

    if schema_failures:
        details = [r.to_dict() for r in schema_results]
        _log_run(engine, run_id, table_name, STATUS_FAILED,
                 dq_passed=len(schema_results) - len(schema_failures),
                 dq_failed=len(schema_failures),
                 dq_details=details, file_hash=file_hash,
                 duration=time.time() - start,
                 error_msg="Schema validation failed")
        return {"table": table_name, "status": STATUS_FAILED, "error": "Schema validation failed",
                "details": details}

    total_rows = 0
    try:
        chunks = pd.read_csv(source_file, chunksize=READ_CHUNKSIZE)
        first = True
        for chunk in chunks:
            chunk.to_sql(
                table_name, engine,
                if_exists="replace" if first else "append",
                index=False, chunksize=WRITE_CHUNKSIZE,
            )
            first = False
            total_rows += len(chunk)
    except Exception as e:
        _log_run(engine, run_id, table_name, STATUS_FAILED,
                 rows_loaded=total_rows, file_hash=file_hash,
                 duration=time.time() - start, error_msg=str(e))
        return {"table": table_name, "status": STATUS_FAILED, "error": str(e)}

    loaded_df = pd.read_sql(f"SELECT * FROM {table_name}", engine)
    dq_results = run_dq_checks(loaded_df, table_config)
    dq_passed = sum(1 for r in dq_results if r.passed)
    dq_failed = sum(1 for r in dq_results if not r.passed)
    all_details = [r.to_dict() for r in schema_results + dq_results]

    final_status = STATUS_SUCCESS if dq_failed == 0 else STATUS_SUCCESS

    _log_run(engine, run_id, table_name, final_status,
             rows_loaded=total_rows, rows_failed=0,
             dq_passed=dq_passed + len(schema_results),
             dq_failed=dq_failed,
             dq_details=all_details, file_hash=file_hash,
             duration=time.time() - start)

    return {
        "table": table_name,
        "status": final_status,
        "rows_loaded": total_rows,
        "dq_passed": dq_passed,
        "dq_failed": dq_failed,
        "duration": round(time.time() - start, 2),
        "details": all_details,
    }


def run_pipeline(force: bool = False) -> list[dict]:
    """
    Run the full ETL pipeline for all tables.
    Returns a list of per-table summary dicts.
    """
    os.makedirs("database", exist_ok=True)
    data_dir = _resolve_data_dir()
    engine = _get_engine()
    _init_pipeline_runs_table(engine)

    results = []
    for table_name, config in TABLE_CONFIGS.items():
        print(f"[Pipeline] Processing {table_name}...")
        result = ingest_table(engine, table_name, config, data_dir, force=force)
        print(f"  -> {result['status']} ({result.get('rows_loaded', 0):,} rows)")
        results.append(result)

    for table_name, config in TABLE_CONFIGS.items():
        fk = config.get("foreign_key")
        if fk:
            ri_result = check_referential_integrity(engine, fk)
            print(f"[Pipeline] Ref integrity {table_name}.{fk['column']}: "
                  f"{'PASS' if ri_result.passed else 'FAIL'} — {ri_result.details}")

    return results


def get_pipeline_history(engine, limit: int = 50) -> pd.DataFrame:
    """Fetch recent pipeline run history."""
    try:
        sql = f"""
        SELECT run_id, run_timestamp, table_name, status,
               rows_loaded, rows_failed,
               dq_checks_passed, dq_checks_failed,
               file_hash, duration_secs, error_message
        FROM pipeline_runs
        ORDER BY run_timestamp DESC
        LIMIT {limit}
        """
        return pd.read_sql(sql, engine)
    except Exception:
        return pd.DataFrame()


def get_pipeline_summary(engine) -> dict:
    """Get high-level pipeline health metrics."""
    try:
        sql = """
        SELECT table_name,
               MAX(run_timestamp) as last_run,
               status,
               rows_loaded,
               dq_checks_passed,
               dq_checks_failed,
               duration_secs
        FROM pipeline_runs
        WHERE run_id IN (
            SELECT run_id FROM (
                SELECT run_id, table_name,
                       ROW_NUMBER() OVER (PARTITION BY table_name ORDER BY run_timestamp DESC) as rn
                FROM pipeline_runs
            ) WHERE rn = 1
        )
        GROUP BY table_name
        """
        latest = pd.read_sql(sql, engine)

        total_sql = """
        SELECT
            COUNT(*) as total_runs,
            SUM(CASE WHEN status='success' THEN 1 ELSE 0 END) as successes,
            SUM(CASE WHEN status='failed' THEN 1 ELSE 0 END) as failures,
            SUM(CASE WHEN status='skipped' THEN 1 ELSE 0 END) as skips
        FROM pipeline_runs
        """
        totals = pd.read_sql(total_sql, engine).iloc[0]

        return {
            "latest_runs": latest,
            "total_runs": int(totals["total_runs"]),
            "successes": int(totals["successes"]),
            "failures": int(totals["failures"]),
            "skips": int(totals["skips"]),
        }
    except Exception:
        return {"latest_runs": pd.DataFrame(), "total_runs": 0,
                "successes": 0, "failures": 0, "skips": 0}


def get_sla_status(engine) -> pd.DataFrame:
    """
    Compute SLA status per table by comparing latest run duration
    to the rolling average of the last 10 successful runs.
    Returns DataFrame with columns: table_name, latest_duration, avg_duration, sla_ratio, sla_status.
    """
    try:
        sql = """
        WITH ranked AS (
            SELECT table_name, duration_secs, run_timestamp,
                   ROW_NUMBER() OVER (PARTITION BY table_name ORDER BY run_timestamp DESC) as rn
            FROM pipeline_runs
            WHERE status = 'success' AND duration_secs IS NOT NULL
        ),
        latest AS (
            SELECT table_name, duration_secs as latest_duration
            FROM ranked WHERE rn = 1
        ),
        rolling AS (
            SELECT table_name, AVG(duration_secs) as avg_duration
            FROM ranked WHERE rn <= 10
            GROUP BY table_name
        )
        SELECT l.table_name,
               ROUND(l.latest_duration, 2) as latest_duration,
               ROUND(r.avg_duration, 2) as avg_duration,
               ROUND(l.latest_duration / r.avg_duration, 2) as sla_ratio
        FROM latest l
        JOIN rolling r ON l.table_name = r.table_name
        """
        df = pd.read_sql(sql, engine)
        if df.empty:
            return df
        df["sla_status"] = df["sla_ratio"].apply(
            lambda x: "BREACH" if x > 2.0 else ("WARNING" if x > 1.5 else "OK")
        )
        return df
    except Exception:
        return pd.DataFrame()


def get_duration_trend(engine, limit: int = 20) -> pd.DataFrame:
    """Get pipeline duration history for trend visualization."""
    try:
        sql = f"""
        SELECT table_name, run_timestamp, duration_secs
        FROM pipeline_runs
        WHERE status = 'success' AND duration_secs IS NOT NULL
        ORDER BY run_timestamp DESC
        LIMIT {limit * 3}
        """
        return pd.read_sql(sql, engine)
    except Exception:
        return pd.DataFrame()


PIPELINE_DAG = {
    "nodes": [
        {"id": "csv_apps", "label": "application_train.csv", "layer": 0, "type": "source"},
        {"id": "csv_bureau", "label": "bureau.csv", "layer": 0, "type": "source"},
        {"id": "csv_prev", "label": "previous_application.csv", "layer": 0, "type": "source"},
        {"id": "hash_check", "label": "Checksum Detection\n(MD5)", "layer": 1, "type": "process"},
        {"id": "schema_val", "label": "Schema Validation\n(columns, dtypes)", "layer": 2, "type": "process"},
        {"id": "load", "label": "Chunked Load\n(10k read / 1k write)", "layer": 3, "type": "process"},
        {"id": "dq_checks", "label": "DQ Checks\n(nulls, ranges, PKs,\nsentinels)", "layer": 4, "type": "process"},
        {"id": "ref_integrity", "label": "Referential Integrity\n(FK validation)", "layer": 5, "type": "process"},
        {"id": "db_apps", "label": "applications\n(SQLite)", "layer": 6, "type": "sink"},
        {"id": "db_bureau", "label": "bureau\n(SQLite)", "layer": 6, "type": "sink"},
        {"id": "db_prev", "label": "previous_application\n(SQLite)", "layer": 6, "type": "sink"},
        {"id": "log", "label": "pipeline_runs\n(run log)", "layer": 6, "type": "sink"},
    ],
    "edges": [
        ("csv_apps", "hash_check"),
        ("csv_bureau", "hash_check"),
        ("csv_prev", "hash_check"),
        ("hash_check", "schema_val"),
        ("schema_val", "load"),
        ("load", "dq_checks"),
        ("dq_checks", "ref_integrity"),
        ("ref_integrity", "db_apps"),
        ("ref_integrity", "db_bureau"),
        ("ref_integrity", "db_prev"),
        ("dq_checks", "log"),
    ],
}


if __name__ == "__main__":
    results = run_pipeline(force=True)
    print("\n=== Pipeline Summary ===")
    for r in results:
        print(f"  {r['table']}: {r['status']} — {r.get('rows_loaded', 0):,} rows "
              f"({r.get('dq_passed', 0)} checks passed, {r.get('dq_failed', 0)} failed)")