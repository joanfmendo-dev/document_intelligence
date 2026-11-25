FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml ./
COPY docintel ./docintel
RUN pip install --no-cache-dir ".[azure]" \
 && python -c "from rapidocr import RapidOCR; RapidOCR()"
# no secrets in the image: USERS, SECRET_KEY, the *_API_KEY variables and BLOB_CONNECTION_STRING arrive as environment variables
ENV ALLOWED_HOSTS=* DB_PATH=/data/medical_records.duckdb UPLOADS_DIR=/data/uploads CONFIG_FILE=/nonexistent
EXPOSE 8000
CMD ["python", "-m", "docintel.web", "--host", "0.0.0.0", "--port", "8000"]
