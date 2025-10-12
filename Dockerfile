# Dockerfile for backup inference server
# Uses Hugging Face Inference API if credits are available, otherwise falls back to Ollama

FROM python:3.12-slim

# Install dependencies
RUN pip install --no-cache-dir huggingface_hub gradio requests

# Install Ollama (official method)
RUN apt-get update && apt-get install -y curl && \
    curl -fsSL https://ollama.com/install.sh | bash

# Copy server script
COPY backup_server.py /app/backup_server.py
WORKDIR /app


# Expose port for Gradio
EXPOSE 5000

# Entrypoint
CMD ["python", "backup_server.py"]
