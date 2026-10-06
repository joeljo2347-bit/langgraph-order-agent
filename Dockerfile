FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY order_agent order_agent
ENV OLLAMA_HOST=http://host.docker.internal:11434 \
    ORDER_AGENT_MODEL=ollama:gpt-oss:20b
EXPOSE 8000
USER nobody
CMD ["uvicorn", "order_agent.api:app", "--host", "0.0.0.0", "--port", "8000"]
