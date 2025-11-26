# Hướng dẫn chạy dự án Predict Number

Dự án cung cấp pipeline dự đoán xổ số Mega 6/45 với các thành phần FastAPI, Streamlit dashboard, Kafka/Faust streaming, Redis và Airflow DAG. Tài liệu này hướng dẫn cài đặt, huấn luyện, suy luận và chạy toàn bộ stack.

## 1. Yêu cầu hệ thống
- Python 3.12 trở lên.
- `uv` (khuyến nghị) hoặc `pip` để cài đặt phụ thuộc.
- Docker + Docker Compose (nếu muốn chạy toàn bộ stack streaming).
- Kafka, Redis sẽ được dựng sẵn khi chạy Docker Compose.

## 2. Cấu trúc thư mục chính
- `data/` – dữ liệu đầu vào (`multi_hot_matrix.csv`) và đầu ra (`data/output`).
- `models/` – nơi lưu các phiên bản mô hình và `registry.json`.
- `src/core_model/` – logic ML: tiền xử lý, đặc trưng, huấn luyện, suy luận, registry, CLI predict tiếp theo.
- `src/api/app.py` – FastAPI phục vụ REST API.
- `src/utils/build_features_parquet.py` – tiện ích xây dựng đặc trưng từ data lake.
- `src/streaming/` – worker Kafka/Faust cho inference thời gian thực.
- `src/storage/` – sink lưu kết quả vào Redis/Parquet.
- `src/ingestion/` – generator & scraper stub đưa dữ liệu vào Kafka.
- `src/dashboard/app.py` – Dashboard Streamlit đọc kết quả.
- `docker-compose.yml` – khởi chạy Kafka/Redis/API/dashboard/worker bằng Docker.
- `pyproject.toml` – khai báo script tiện dụng cho `uv run` (`api`, `train`, `predict_next`, `build_features`, `build-features`).

## 3. Cài đặt phụ thuộc
### Cách 1: Dùng `uv` (khuyến nghị)
```bash
uv pip install -r requirements.txt
```

### Cách 2: Dùng `pip`
```bash
python -m venv .venv
source .venv/bin/activate  # hoặc .venv\\Scripts\\activate trên Windows
pip install -r requirements.txt
```

> Lưu ý: Nếu `uv` báo lỗi không nhận `[project.scripts]`, hãy đảm bảo đang dùng phiên bản `uv` mới (đã hỗ trợ mục này trong `pyproject.toml`).

## 4. Chuẩn bị dữ liệu
- File mặc định: `data/multi_hot_matrix.csv`. Đặt file này đúng vị trí trước khi huấn luyện hoặc suy luận.
- Các script tự chuyển đổi đường dẫn tương đối sang tuyệt đối nên có thể chạy từ thư mục gốc repo.

## 5. Huấn luyện mô hình & tạo version mới
Chạy từ thư mục gốc repo để `uv` tìm thấy `pyproject.toml`:
```bash
# Cách 1: uv run --script (ổn định nhất trên Windows)
uv run --script train

# Cách 2: uv run (nếu bản uv hỗ trợ gọi trực tiếp script name)
uv run train

# Fallback rõ ràng nếu vẫn báo "program not found"
uv run -- python src/core_model/train.py data/multi_hot_matrix.csv
python src/core_model/train.py data/multi_hot_matrix.csv
```
Script sẽ:
- Nạp dữ liệu CSV, xây đặc trưng nâng cao.
- Huấn luyện 6 mô hình logistic (n_1…n_6).
- Tạo thư mục phiên bản `models/vYYYYMMDD_HHMMSS/` chứa model & scaler.
- Ghi/ghi đè `models/registry.json` và đặt `current_version` = version mới.

## 6. Dự đoán batch kỳ quay tiếp theo
```bash
# Ưu tiên: uv với --script
uv run --script predict_next

# Hoặc (nếu bản uv hỗ trợ):
uv run predict_next

# Fallback
uv run -- python src/core_model/inference_next.py --data data/multi_hot_matrix.csv --output data/output/next_prediction.json
python src/core_model/inference_next.py --data data/multi_hot_matrix.csv --output data/output/next_prediction.json
```
Kết quả sẽ được in ra console và lưu thành JSON tại `data/output/next_prediction.json` (tạo thư mục nếu chưa có).

## 7. Xây dựng đặc trưng từ data lake (tuỳ chọn)
```bash
# Ổn định nhất (uv run --script)
uv run --script build_features
uv run --script build-features

# Nếu uv hỗ trợ gọi trực tiếp
uv run build_features
uv run build-features

# Fallback Windows khi uv không nhận script name
python build_features.py
python src/utils/build_features_parquet.py

# Hoặc chạy file batch (Windows) đi kèm
build_features.bat
build-features.bat
```
Script đọc data lake và tạo các file Parquet đặc trưng phục vụ huấn luyện/batch.

## 8. Chạy API FastAPI (cục bộ)
```bash
# Ổn định: dùng uv run --script
uv run --script api

# Nếu bản uv hỗ trợ gọi trực tiếp
uv run api

# Fallback
uvicorn src.api.app:app --reload --port 8000
```
Các endpoint chính:
- `GET /status` – kiểm tra health.
- `POST /predict` – nhận JSON `{ "features": [...] }` và trả dự đoán cho bộ đặc trưng đã chuẩn hoá.
- `GET /predict_next_draw` – dùng dữ liệu lịch sử để dự đoán kỳ tiếp theo (yêu cầu file CSV tồn tại).
- `GET /prediction/{id}` – lấy kết quả lưu trong Redis (khi chạy kèm pipeline streaming/prediction sink).

## 9. Chạy toàn bộ stack bằng Docker Compose
```bash
docker-compose up --build
```
Dịch vụ chính & cổng:
- FastAPI: `http://localhost:8000`
- Streamlit Dashboard: `http://localhost:8501`
- Airflow Web UI: `http://localhost:8080` (admin/admin)
- Kafka broker: `localhost:9092`
- Redis: `localhost:6379`

Luồng mặc định:
1. `generator` sinh dữ liệu vào Kafka.
2. `consumer` lấy feature → suy luận → đẩy topic `predictions`.
3. `prediction-sink` đọc `predictions`, lưu Redis + Parquet.
4. Dashboard & API đọc từ Redis/Data Lake để hiển thị/kết quả.

## 10. Mẹo xử lý sự cố
- **`uv run train` báo "program not found"**: chạy `uv run train`; nếu vẫn lỗi, dùng `uv run -- python src/core_model/train.py ...`.
- **Thiếu file dữ liệu**: chắc chắn `data/multi_hot_matrix.csv` tồn tại; API `/predict_next_draw` sẽ báo lỗi 500 nếu thiếu.
- **Registry trống/thiếu version**: chạy lại bước huấn luyện để tạo `models/registry.json` và version mới.
- **Path trên Windows**: ưu tiên chạy từ thư mục gốc repo và dùng `uv run --script ...` để tránh lỗi PYTHONPATH.
- ** Nếu Airflow chưa hiển thị DAG**: docker exec -d airflow airflow scheduler để khởi động scheduler.

## 11. Thành phần bổ trợ (tham khảo)
- **Streaming/Faust**: `src/streaming/consumer_worker.py`, `src/streaming/faust_worker.py` xử lý message Kafka và suy luận thời gian thực.
- **Storage sinks**: `src/storage/prediction_sink.py`, `src/storage/parquet_sink.py` lưu kết quả vào Redis/Parquet.
- **Ingestion**: `src/ingestion/data_generator.py` và `src/ingestion/scraper_job.py` tạo/gom dữ liệu vào Kafka.
- **Airflow DAG**: `airflow/dags/get_data_raw/retrain_predict_model.py` minh hoạ pipeline scraping → feature → train → predict.

## 12. Kiểm thử nhanh
Chạy kiểm tra cú pháp tối thiểu:
```bash
python -m compileall src/core_model/train.py src/core_model/inference_next.py src/api/app.py
```