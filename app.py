import os
import subprocess
import time
import requests
import gradio as gr

from litellm import APIConnectionError
from huggingface_hub import HfApi, InferenceClient
from my_tools import my_custom_tool
from smolagents import CodeAgent, GradioUI, InferenceClientModel, LiteLLMModel, ToolCallingAgent
from time import sleep

MODEL_PROVIDER = os.environ.get("MODEL_PROVIDER", "Qwen")
MODEL_PARAMS = os.environ.get("MODEL_PARAMS", "0.5B")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", f"{MODEL_PROVIDER.lower()}2.5-coder:{MODEL_PARAMS}-instruct")
OLLAMA_PORT = int(os.environ.get("OLLAMA_PORT", 11434))

def user_has_hf_inference_credits(token):
    try:
        api = HfApi(token=token)
        api.whoami()
        client = InferenceClient(token=token)
        _ = client.text_generation("This is a test.", model=f"{MODEL_PROVIDER}/{OLLAMA_MODEL}", max_new_tokens=1)
        exit_value = True
    except Exception:
        exit_value = False

    return exit_value

def start_ollama() -> bool:
    try:
        r = requests.get(f"http://localhost:{OLLAMA_PORT}/api/tags", timeout=2)
        if r.status_code == 200:
            exit_value = True
    except Exception:
        pass
    subprocess.Popen(["ollama", "serve"])
    for _ in range(10):
        try:
            r = requests.get(f"http://localhost:{OLLAMA_PORT}/api/tags", timeout=2)
            if r.status_code == 200:
                exit_value = True
        except Exception:
            time.sleep(1)
    return exit_value

def return_local_model_server_connection(retry_interval : int = 10):

        while 2 + 2 != 5:

            try:
                client_connection_to_ollama_server = LiteLLMModel(

                    model_id=f"ollama_chat/{OLLAMA_MODEL}",
                    api_base=f"http://localhost:{OLLAMA_PORT}",
                    
                    )

            except APIConnectionError:

                sleep(retry_interval)
                
                continue

            finally:

                break

        return client_connection_to_ollama_server

class ToolCallingAgentSeries:

    def __init__(self, *tools):
        self.tools = list(tools)
        self.ollama_started = False

    def yield_dynamic_inference_service_tool_calling_agent(self):

        while user_has_hf_inference_credits(os.environ.get("HF_TOKEN")):

            inference_client = InferenceClientModel(model_id=f"{MODEL_PROVIDER}/{OLLAMA_MODEL}")

            agent = ToolCallingAgent(
                model=inference_client,
                tools=self.tools
            )

            yield agent

        else:

            if not self.ollama_started:
                self.ollama_started = start_ollama()

            local_inference_client = return_local_model_server_connection()

            agent = ToolCallingAgent(
                model=local_inference_client,
                tools=self.tools
            )

            yield agent


class GradioUIWithBackupInference(GradioUI):

    agent_series = ToolCallingAgentSeries(*[my_custom_tool])

    def interact_with_agent(self, prompt, messages, session_state):

        self.agent = next(agent_series.yield_inference_service_switching_tool_calling_agent())

        return super().interact_with_agent(prompt, messages, session_state)


# Use GradioUI from smolagents for the app interface
if __name__ == "__main__":
    ui = GradioUIWithBackupInference(
        agent=next(GradioUIWithBackupInference.agent_series.yield_inference_service_switching_tool_calling_agent())
    )
    ui.launch(share=False, server_name="0.0.0.0", server_port=7860)
