# Pre-pulls BASE_LANGUAGE_MODEL in the event the application needs to fall back to Ollama.

FROM python:3.12-slim

ARG LANGUAGE_MODEL_PROVIDER=qwen

ARG LANGUAGE_MODEL_VERSION=2.5

ARG LANGUAGE_MODEL_PARAMETERS=7b

ENV BASE_LANGUAGE_MODEL=${LANGUAGE_MODEL_PROVIDER}${LANGUAGE_MODEL_VERSION}-coder:${LANGUAGE_MODEL_PARAMETERS}-instruct

# Install Ollama (official method)
RUN apt-get update && apt-get install -y curl && \
    curl -fsSL https://ollama.com/install.sh | bash

RUN apt-get update && apt-get install -y procps && rm -rf /var/lib/apt/lists/*

RUN ollama serve & sleep 5 ; \
    ollama pull $BASE_LANGUAGE_MODEL ; \
    echo "kill 'ollama serve' process" ; \
    ps -ef | grep 'ollama serve' | grep -v grep | awk '{print $2}' | xargs -r kill -9
    

# Copy server script
COPY app.py /app/app.py
COPY requirements.txt /app/requirements.txt
WORKDIR /app

RUN pip install --no-cache-dir -r requirements.txt

# Expose port for Gradio
EXPOSE 7860

# Entrypoint
CMD ["python", "app.py"]
