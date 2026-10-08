FROM python:3.11-slim
WORKDIR /app

# CPU-only torch keeps the image far smaller than the default GPU build
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 8000 8501
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]