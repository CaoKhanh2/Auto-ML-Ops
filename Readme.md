\# Predict Number Big Data Realtime



\## 1. Chuẩn bị



\- Python 3.11

\- Docker + Docker Compose



Cấu trúc chính:



\- `data/multi\_hot\_matrix.csv` – dữ liệu lịch sử (từ project gốc)

\- `src/core\_model/\*` – logic ML

\- `src/streaming/faust\_worker.py` – streaming inference

\- `src/ingestion/data\_generator.py` – generator -> Kafka

\- `src/api/app.py` – FastAPI API

\- `src/storage/\*` – Redis + Data Lake

\- `docker-compose.yml`, `Dockerfile`, `requirements.txt`



\## 2. Train model \& tạo version



```bash

\# chạy local

PYTHONPATH=src python -m core\_model.train data/multi\_hot\_matrix.csv



