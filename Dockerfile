FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

ENV DATASET_PATH=/app/data/abaDatasetV1.csv
ENV OLLAMA_URL=http://localhost:11434/api/generate
ENV LLM_MODEL=llama3

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
