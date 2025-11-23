from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

PROJECT_DIR = "/app"

SCRAPER = f"{PROJECT_DIR}/src/ingestion/getData.py"
BUILD_FEATURES = f"{PROJECT_DIR}/src/utils/build_features_parquet.py"
BUILD_MULTI_HOT = f"{PROJECT_DIR}/src/utils/build_multi_hot_from_lake.py"
TRAIN_MODEL = f"{PROJECT_DIR}/src/core_model/train.py"
PREDICT_NEXT = f"{PROJECT_DIR}/src/core_model/inference_next.py"

default_args = {
    "owner": "auto-mlops",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=3),
}

with DAG(
    dag_id="mega645_auto_pipeline",
    default_args=default_args,
    description="Full Auto MLOps pipeline for Mega 6/45",
    schedule_interval="5 18 * * WED,FRI,SUN",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["mega645", "mlops", "automl"],
) as dag:

    ingest_task = BashOperator(
        task_id="ingest_new_results",
        bash_command=f"python {SCRAPER} --auto",
    )

    build_features = BashOperator(
        task_id="build_features",
        bash_command=f"python {BUILD_FEATURES}",
    )

    build_multi_hot = BashOperator(
        task_id="build_multi_hot",
        bash_command=f"python {BUILD_MULTI_HOT}",
    )

    train_models = BashOperator(
        task_id="train_model",
        bash_command=f"python {TRAIN_MODEL} data/multi_hot_matrix.csv",
    )

    predict_next = BashOperator(
        task_id="predict_next_draw",
        bash_command=f"python {PREDICT_NEXT}",
    )

    ingest_task >> build_features >> build_multi_hot >> train_models >> predict_next
