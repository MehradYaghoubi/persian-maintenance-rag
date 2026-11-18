FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt requirements-semantic.txt ./
RUN pip install --no-cache-dir -r requirements-semantic.txt
COPY app.py retrieval.py semantic.py evaluate.py demo.html ./
COPY manuals ./manuals
COPY evaluation ./evaluation
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
