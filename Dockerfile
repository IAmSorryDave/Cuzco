# Dockerfile for backup inference server
# Uses Hugging Face Inference API if credits are available, otherwise falls back to Ollama

FROM python:3.12-slim

# Install Ollama (official method)
RUN apt-get update && apt-get install -y curl && \
    curl -fsSL https://ollama.com/install.sh | bash

ENV OLLAMA_MODEL=qwen2.5-coder:0.5b-instruct

RUN ollama serve & sleep 5 ; ollama pull $OLLAMA_MODEL ; echo "kill 'ollama serve' process" ; ps -ef | grep 'ollama serve' | grep -v grep | awk '{print $2}' | xargs -r kill -9

# Copy server script
COPY backup_server.py /app/backup_server.py
COPY requirments.txt /app/requirments.txt
WORKDIR /app

RUN pip install --no-cache-dir -r requirments.txt

# Expose port for Gradio
EXPOSE 5000

# Entrypoint
CMD ["python", "backup_server.py"]
