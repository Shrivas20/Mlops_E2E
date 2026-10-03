FROM python:3.10-slim-bookworm

WORKDIR /app

COPY . /app

ENV PYTHONPATH=/app/src

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "app.py"]