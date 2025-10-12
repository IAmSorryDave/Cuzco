
# Cuzco
Backup Inference Server for Hugging Face Spaces

## Usage

1. Build the Docker image:

	docker build -t backup-inference-server .

2. Run the container with your Hugging Face token:

	docker run -e HF_TOKEN=your_hf_token -p 5000:5000 backup-inference-server

- The server exposes a POST endpoint at `/infer` with JSON body `{ "prompt": "your prompt here" }`.
- If Hugging Face inference credits are available, it uses Hugging Face. If not, it falls back to Ollama.

## Environment Variables
- `HF_TOKEN`: Your Hugging Face API token (required for Hugging Face inference)
- `OLLAMA_MODEL`: (optional) Model to use with Ollama (default: llama2)
- `OLLAMA_PORT`: (optional) Ollama server port (default: 11434)

## Example Request

```
curl -X POST http://localhost:5000/infer \
	  -H 'Content-Type: application/json' \
	  -d '{"prompt": "What is the capital of Peru?"}'
```
