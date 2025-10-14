import os
import subprocess
import requests

from collections.abc import Generator
from litellm import APIConnectionError
from huggingface_hub import HfApi, InferenceClient
from smolagents import CodeAgent, GradioUI, InferenceClientModel, LiteLLMModel, CodeAgent, WebSearchTool, FinalAnswerTool, ToolCallingAgent
from time import sleep

from typing import Generator

LANGUAGE_MODEL_PARAMETERS = os.environ.get("LANGUAGE_MODEL_PARAMETERS", "3b")
LANGUAGE_MODEL_PROVIDER = os.environ.get("LANGUAGE_MODEL_PROVIDER", "qwen")
LANGUAGE_MODEL_VERSION = os.environ.get("LANGUAGE_MODEL_VERSION", 2.5)
OLLAMA_MODEL = os.environ.get(
    "BASE_LANGUAGE_MODEL", 
    f"{LANGUAGE_MODEL_PROVIDER.title()}{LANGUAGE_MODEL_VERSION}-coder:{LANGUAGE_MODEL_PARAMETERS}-instruct"
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

    AgentType = ToolCallingAgent

    stream_outputs = True

    planning_interval = 2

    def __init__(self, *tools):
        self.tools = list(tools)
        self.__agents_point_to_ollama = False

    @property
    def agents_point_to_ollama(self):
        return self.__agents_point_to_ollama
        
    @agents_point_to_ollama.setter
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

    agent_series = LanguageModelAgentGenerator(*[WebSearchTool(), FinalAnswerTool()])

    def refresh_agent(self) -> None:
        self.agent = next(self.agent_series)
     
    def create_app(self):
        import gradio as gr

        with gr.Blocks(theme="ocean", fill_height=True) as demo:
            # Add session state to store session-specific data
            session_state = gr.State({})
            stored_messages = gr.State([])
            file_uploads_log = gr.State([])

            with gr.Sidebar():
                gr.Markdown(
                    f"# {self.name.replace('_', ' ').capitalize()}"
                    "\n> This web ui allows you to interact with a `smolagents` agent that can use tools and execute steps to complete tasks."
                    + (f"\n\n**Agent description:**\n{self.description}" if self.description else "")
                )

                with gr.Group():
                    gr.Markdown("**Your request**", container=True)
                    text_input = gr.Textbox(
                        lines=3,
                        label="Chat Message",
                        container=False,
                        placeholder="Enter your prompt here and press Shift+Enter or press the button",
                    )
                    submit_btn = gr.Button("Submit", variant="primary")

                # If an upload folder is provided, enable the upload feature
                if self.file_upload_folder is not None:
                    upload_file = gr.File(label="Upload a file")
                    upload_status = gr.Textbox(label="Upload Status", interactive=False, visible=False)
                    upload_file.change(
                        self.upload_file,
                        [upload_file, file_uploads_log],
                        [upload_status, file_uploads_log],
                    )

                gr.HTML(
                    "<br><br><h4><center>Powered by <a target='_blank' href='https://github.com/huggingface/smolagents'><b>smolagents</b></a></center></h4>"
                )

            # Main chat interface
            chatbot = gr.Chatbot(
                label="Agent",
                type="messages",
                avatar_images=(
                    None,
                    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/smolagents/mascot_smol.png",
                ),
                resizeable=True,
                scale=1,
                latex_delimiters=[
                    {"left": r"$$", "right": r"$$", "display": True},
                    {"left": r"$", "right": r"$", "display": False},
                    {"left": r"\[", "right": r"\]", "display": True},
                    {"left": r"\(", "right": r"\)", "display": False},
                ],
            )

            # Set up event handlers
            text_input.submit(
                self.log_user_message,
                [text_input, file_uploads_log],
                [stored_messages, text_input, submit_btn],
            ).success(self.refresh_agent).success(self.interact_with_agent, [stored_messages, chatbot, session_state], [chatbot]).then(
                lambda: (
                    gr.Textbox(
                        interactive=True, placeholder="Enter your prompt here and press Shift+Enter or the button"
                    ),
                    gr.Button(interactive=True),
                ),
                None,
                [text_input, submit_btn],
            )

            submit_btn.click(
                self.log_user_message,
                [text_input, file_uploads_log],
                [stored_messages, text_input, submit_btn],
            ).success(self.refresh_agent).success(self.interact_with_agent, [stored_messages, chatbot, session_state], [chatbot]).then(
                lambda: (
                    gr.Textbox(
                        interactive=True, placeholder="Enter your prompt here and press Shift+Enter or the button"
                    ),
                    gr.Button(interactive=True),
                ),
                None,
                [text_input, submit_btn],
            )

            chatbot.clear(self.agent.memory.reset)
        return demo

# Use GradioUI from smolagents for the app interface
if __name__ == "__main__":

    first_agent = next(GradioUIWithBackupInference.agent_series)

    ui = GradioUIWithBackupInference(
        agent=first_agent
    )

    ui.launch(share=False, server_name="0.0.0.0", server_port=7860)


