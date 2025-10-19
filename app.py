import os
import logging
import subprocess
import requests
import huggingface_hub

from collections.abc import Generator
from litellm import APIConnectionError
from huggingface_hub import InferenceClient
from smolagents import CodeAgent, GradioUI, InferenceClientModel, LiteLLMModel
from time import sleep
from traceback import format_exc
from warnings import warn

from typing import Generator

LANGUAGE_MODEL_PARAMETERS = os.environ.get("LANGUAGE_MODEL_PARAMETERS", "7b")
LANGUAGE_MODEL_PROVIDER = os.environ.get("LANGUAGE_MODEL_PROVIDER", "qwen")
LANGUAGE_MODEL_VERSION = os.environ.get("LANGUAGE_MODEL_VERSION", 2.5)

LANGUAGE_MODEL = f"{LANGUAGE_MODEL_PROVIDER}{LANGUAGE_MODEL_VERSION}-coder:{LANGUAGE_MODEL_PARAMETERS}-instruct"

OLLAMA_MODEL = os.environ.get(
    "BASE_LANGUAGE_MODEL", 
    LANGUAGE_MODEL
)

OLLAMA_PORT = int(os.environ.get("OLLAMA_PORT", 11434))

HUGGING_FACE_MODEL_ID = f"{LANGUAGE_MODEL_PROVIDER.title()}/{LANGUAGE_MODEL.title().replace(':','-')}"


def user_can_login_to_hf(token : str) -> bool:
    try:
        huggingface_hub.login(token=token)
        exit_value = True
    except Exception as e:
        logging.error(format_exc())
        exit_value = False
    finally:
        return exit_value

def ping_hf_inference(token : str) -> None:

    agent = CodeAgent(  
        tools=list(),
        max_steps=1,
        model=InferenceClientModel(model_id=HUGGING_FACE_MODEL_ID, token=token),
        planning_interval=1,
        max_print_outputs_length=0
    )

    agent.run("_", return_full_result = False) 
    
    agent.interrupt()

def user_has_hugging_face_inference_credits(token : str) -> bool:

    exit_value = True
    
    try:
 
        ping_hf_inference(token)
        
    except requests.exceptions.HTTPError as err:
        if err.response.status_code == 402:
            exit_value = False

    except Exception as e:
        logging.error(format_exc())
        
    finally:
        return exit_value

def check_if_user_can_login_to_hf_and_has_hf_inference_credits(token : str) -> bool:
    print("Checking if user can log into Hugging Face and has inference credits")
    return user_can_login_to_hf(token) & user_has_hugging_face_inference_credits(token)

def ping_ollama_server(host = "http://localhost", port = OLLAMA_PORT, timeout = 2):
    return requests.get(f"{host}:{port}/api/tags", timeout=timeout)

def start_local_ollama_server() -> bool:
    try:
        ping = ping_ollama_server()
        if ping.status_code == 200:
            exit_value = True
    except Exception as e:
        logging.error(format_exc())
        exit_value = False
        subprocess.Popen(["ollama", "serve"])
        for _ in range(10):
            try:
                ping = ping_ollama_server()
                if ping.status_code == 200:
                    exit_value = True
                    break
            except Exception as e:
                logging.error(format_exc())
                sleep(1)
    finally:

        if not exit_value:
            warn("Failed to start local Ollama server.")
        
        return exit_value

def return_ollama_server_client_connection(host = "http://localhost", model = OLLAMA_MODEL, port = OLLAMA_PORT, retry_interval : int = 10):

        while 2 + 2 != 5:

            try:
                language_model_client_connection_to_ollama_server = LiteLLMModel(

                    model_id=f"ollama_chat/{model}",
                    api_base=f"{host}:{port}",
                    
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

    add_base_tools = True

    def __init__(self, *initial_tools):
        super().__init__()
        self.tools = list(initial_tools)
        self.__agents_point_to_ollama = False

    @property
    def agents_point_to_ollama(self):
        return self.__agents_point_to_ollama
        
    @agents_point_to_ollama.setter
    def agents_point_to_ollama(self, new_boolean):
        if new_boolean:
            self.__agents_point_to_ollama = new_boolean # Once agents point to Ollama, it should stay that way.
        else:
            warn("Ignoring setting of attribute 'agents_point_to_ollama'. Once agents point to Ollama, it should stay that way until application is restarted.")
        
    def throw(self, type=None, value=None, traceback=None):
        raise StopIteration
        
    def send(self, _):

        if check_if_user_can_login_to_hf_and_has_hf_inference_credits(os.environ.get("HF_TOKEN", '')):

            inference_client = InferenceClientModel(model_id=HUGGING_FACE_MODEL_ID)

            agent = self.AgentType(
                model=inference_client,
                tools=self.tools,
                planning_interval=self.planning_interval,
                stream_outputs=self.stream_outputs,
                add_base_tools=self.add_base_tools
            )

        else:

            if not self.agents_point_to_ollama:
                self.agents_point_to_ollama = start_local_ollama_server()

            local_inference_client = return_ollama_server_client_connection()

            agent = self.AgentType(
                model=local_inference_client,
                tools=self.tools,
                planning_interval=self.planning_interval,
                stream_outputs=self.stream_outputs,
                add_base_tools=self.add_base_tools
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
