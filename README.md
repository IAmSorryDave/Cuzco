
# Cuzco
A platform invariant chatbot template.

## Usage in Codespace

1. Build the Docker image:

	bash up.sh

2. Shutdown the container:

	bash down.sh

## Usage in HF Space

Copy Dockerfile and app.py into HF Space files.
	
- If Hugging Face inference credits are available, it uses Hugging Face. If not, it falls back to Ollama.
- Be sure to configure `HF_TOKEN`: Your Hugging Face API token (required for Hugging Face inference), in your Hugging Face Space if you want to use the HF inference endpoint. Failing to do so will casue inference to defualt to Ollama.

## Environment Variables
- `HF_TOKEN`: Your Hugging Face API token (required for Hugging Face inference)
- `OLLAMA_MODEL`: (optional) Model to use with Ollama (default: llama2)
- `OLLAMA_PORT`: (optional) Ollama server port (default: 11434)
