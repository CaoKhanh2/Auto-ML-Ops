import os
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

PROJECT_DIR = os.getenv("PROJECT_DIR", "/app")

# Định nghĩa đường dẫn
SCRAPER = f"{PROJECT_DIR}/src/ingestion/getData.py"
BUILD_FEATURES = f"{PROJECT_DIR}/src/utils/build_features_parquet.py"
BUILD_MULTI_HOT = f"{PROJECT_DIR}/src/utils/build_multi_hot_from_lake.py"
TRAIN_MODEL = f"{PROJECT_DIR}/src/core_model/train.py"
EVALUATE_MODEL = f"{PROJECT_DIR}/src/utils/evaluate_models.py"  # <--- Thêm script này
PREDICT_NEXT = f"{PROJECT_DIR}/src/core_model/inference_next.py"

default_args = {
    "owner": "auto-mlops",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
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
        bash_command=f"python {SCRAPER}",
        cwd=PROJECT_DIR,
    )

    build_features = BashOperator(
        task_id="build_features",
        bash_command=f"python {BUILD_FEATURES}",
        cwd=PROJECT_DIR,
    )

    build_multi_hot = BashOperator(
        task_id="build_multi_hot",
        bash_command=f"python {BUILD_MULTI_HOT}",
        cwd=PROJECT_DIR,
    )

    train_models = BashOperator(
        task_id="train_model",
        bash_command=f"python {TRAIN_MODEL} {PROJECT_DIR}/data/multi_hot_matrix.csv",
        cwd=PROJECT_DIR,
    )

    # --- Task Mới: Đánh giá mô hình ---
    evaluate_models = BashOperator(
        task_id="evaluate_models",
        bash_command=f"python {EVALUATE_MODEL}",
        cwd=PROJECT_DIR,
    )

    predict_next = BashOperator(
        task_id="predict_next_draw",
        bash_command=f"python {PREDICT_NEXT} --data {PROJECT_DIR}/data/multi_hot_matrix.csv --output {PROJECT_DIR}/data/output/next_prediction.json",
        cwd=PROJECT_DIR,
    )

    # Cập nhật luồng chạy: Train xong -> Evaluate -> Predict Next
    ingest_task >> build_features >> build_multi_hot >> train_models >> evaluate_models >> predict_next