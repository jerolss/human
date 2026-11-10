FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV HUMAN_HOST=0.0.0.0 PORT=8080
EXPOSE 8080
# Read once at boot so a fresh container has something to talk about.
CMD ["sh", "-c", "python -m human.ingest --limit 10 || true; python -m human.server"]
