import os
import subprocess
import requests

from collections.abc import Generator
from litellm import APIConnectionError
from huggingface_hub import HfApi, InferenceClient
from my_tools import my_custom_tool
from smolagents import CodeAgent, GradioUI, InferenceClientModel, LiteLLMModel, CodeAgent
from time import sleep

LANGUAGE_MODEL_PARAMETERS = os.environ.get("LANGUAGE_MODEL_PARAMETERS", "3b")
LANGUAGE_MODEL_PROVIDER = os.environ.get("LANGUAGE_MODEL_PROVIDER", "qwen")
LANGUAGE_MODEL_VERSION = os.environ.get("LANGUAGE_MODEL_VERSION", 2.5)
OLLAMA_MODEL = os.environ.get(
    "BASE_LANGUAGE_MODEL", 
    f"{LANGUAGE_MODEL_PROVIDER.title()}{LANGUAGE_MODEL_VERSION}-coder:{MODEL_PARAMS}-instruct"
)
OLLAMA_PORT = int(os.environ.get("OLLAMA_PORT", 11434))

def user_has_hf_inference_credits(token : str) -> bool:
    try:
        api = HfApi(token=token)
        api.whoami()
        client = InferenceClient(token=token)
        _ = client.text_generation("This is a test.", model=f"{LANGUAGE_MODEL_PROVIDER}/{OLLAMA_MODEL}", max_new_tokens=1)
        exit_value = True
    except Exception:
        exit_value = False
    finally:
        return exit_value

def ping_local_ollama_server():
    return requests.get(f"http://localhost:{OLLAMA_PORT}/api/tags", timeout=2)

def start_local_ollama_server() -> bool:
    try:
        ping = ping_local_ollama_server()
        if ping.status_code == 200:
            exit_value = True
    except Exception:
        exit_value = False
        subprocess.Popen(["ollama", "serve"])
        for _ in range(10):
            try:
                ping = ping_local_ollama_server()
                if ping.status_code == 200:
                    exit_value = True
            except Exception:
                sleep(1)
    finally:

        if exit_value == False:
            print("Failed to start Ollama.")
        
        return exit_value

def return_local_ollama_server_connection(retry_interval : int = 10):

        while 2 + 2 != 5:

            try:
                language_model_client_connection_to_ollama_server = LiteLLMModel(

                    model_id=f"ollama_chat/{OLLAMA_MODEL}",
                    api_base=f"http://localhost:{OLLAMA_PORT}",
                    
                    )

            except APIConnectionError:

                sleep(retry_interval)
                
                continue

            finally:

                break

        return language_model_client_connection_to_ollama_server

class LanguageModelAgentGenerator(Generator):

    AgentType = CodeAgent

    stream_outputs = True

    planning_interval = 2

    def __init__(self, *tools):
        self.tools = list(tools)
        self.__agents_point_to_ollama = False

    @property
    def agents_point_to_ollama(self):
        return self.__agents_point_to_ollama
        
    @agents.setter
    def agents_point_to_ollama(self, new_boolean):
        if new_boolean:
            self.__agents_point_to_ollama = new_boolean # Once agents point to Ollama, it should stay that way.
        
    def throw(self, type=None, value=None, traceback=None):
        raise StopIteration
        
    def send(self, ignore_me):

        if user_has_hf_inference_credits(os.environ.get("HF_TOKEN")):

            inference_client = InferenceClientModel(model_id=f"{LANGUAGE_MODEL_PROVIDER}/{OLLAMA_MODEL}")

            agent = self.AgentType(
                model=inference_client,
                tools=self.tools,
                planning_interval=self.planning_interval,
                stream_outputs=self.stream_outputs
            )

        else:

            if not self.agents_point_to_ollama:
                self.agents_point_to_ollama = start_local_ollama_server()

            local_inference_client = return_local_ollama_server_connection()

            agent = self.AgentType(
                model=local_inference_client,
                tools=self.tools,
                planning_interval=self.planning_interval,
                stream_outputs=self.stream_outputs
            )

        return agent

class GradioUIWithBackupInference(GradioUI):

    agent_series = LanguageModelAgentGenerator(*[my_custom_tool])

    def interact_with_agent(self, prompt, messages, session_state):
        
        self.agent = next(self.agent_series)

        return super().interact_with_agent(prompt, messages, session_state)


# Use GradioUI from smolagents for the app interface
if __name__ == "__main__":
    agent_series = LanguageModelAgentGenerator(*[my_custom_tool])
    ui = GradioUI(
        agent=next(agent_series)
    )
    ui.launch(share=False, server_name="0.0.0.0", server_port=7860)
