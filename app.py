import os
import logging
import subprocess
import requests

from collections.abc import Generator
from litellm import APIConnectionError
from huggingface_hub import InferenceClient, HfApi
from smolagents import CodeAgent, GradioUI, InferenceClientModel, LiteLLMModel
from time import sleep
from traceback import format_exc
from transformers import AutoTokenizer
from warnings import warn

from typing import Generator, Callable

from huggingface_hub.hf_api import HfFolder

HfFolder.save_token(os.environ.get('HF_ACCESS_TOKEN', ''))

if HfFolder.get_token():
    print("Token")
    os.environ.pop('HF_ACCESS_TOKEN')
else:
    print("No Token")
    HfFolder.delete_token()


LANGUAGE_MODEL_PARAMETERS, LANGUAGE_MODEL_PROVIDER, LANGUAGE_MODEL_VERSION = os.environ.get("LANGUAGE_MODEL_PARAMETERS", "3b"), os.environ.get("LANGUAGE_MODEL_PROVIDER", "qwen"), os.environ.get("LANGUAGE_MODEL_VERSION", 2.5)

LANGUAGE_MODEL = f"{LANGUAGE_MODEL_PROVIDER}{LANGUAGE_MODEL_VERSION}-coder:{LANGUAGE_MODEL_PARAMETERS}-instruct"

OLLAMA_MODEL, OLLAMA_PORT = os.environ.get("BASE_LANGUAGE_MODEL", LANGUAGE_MODEL), int(os.environ.get("OLLAMA_PORT", 11434))

HUGGING_FACE_MODEL_ID = f"{LANGUAGE_MODEL_PROVIDER.title()}/{LANGUAGE_MODEL.title().replace(':','-')}"

def user_is_hf_user() -> bool:
    
    exit_value : bool = False # Assume user is not a Hugging Face user until proven otherwise.

    if HfFolder.get_token():
        try:
            HfApi(token=HfFolder.get_token()).whoami()
            exit_value = True
        except Exception as e:
            logging.error(format_exc())
            exit_value = False
        finally:
            return exit_value
    else:
        return exit_value

def ping_hf_inference() -> None:

    agent = CodeAgent(  
        tools=list(),
        max_steps=1,
        model=InferenceClientModel(model_id=HUGGING_FACE_MODEL_ID, token=HfFolder.get_token()),
        planning_interval=1,
        max_print_outputs_length=0,
        stream_outputs=False,
        add_base_tools=False
    )

    agent.run(AutoTokenizer.from_pretrained(HUGGING_FACE_MODEL_ID).eos_token, return_full_result = False) 
    
    agent.interrupt()

def user_has_hugging_face_inference_credits() -> bool:

    exit_value : bool = False # Assume user does not have credits until proven otherwise.

    if HfFolder.get_token():
    
        try:
    
            ping_hf_inference()

            exit_value = True
            
        except requests.exceptions.HTTPError as err:
            if err.response.status_code == 402:
                exit_value = False
            else:
                logging.error(format_exc())
            
        except Exception as e:
            logging.error(format_exc())

        finally:

            return exit_value

    else:

        return exit_value

def check_if_user_is_hf_user_and_has_hf_inference_credits() -> bool:
    print("Checking if user can log into Hugging Face and has inference credits")
    return user_is_hf_user() & user_has_hugging_face_inference_credits()

def ping_ollama_server(host = "http://localhost", port = OLLAMA_PORT, timeout = 2):
    return requests.get(f"{host}:{port}/api/tags", timeout=timeout)

def start_local_ollama_server() -> bool:

    exit_value : bool = False # Assume server is not running until proven otherwise.

    def inner() -> None:
        nonlocal exit_value
        ping = ping_ollama_server()
        if ping.status_code == 200:
            exit_value = True
        else:
            exit_value = False

    try:

        inner()

    except Exception as e:

        subprocess.Popen(["ollama", "serve"])

        try:

            for _ in range(9):
                inner()

        except Exception as e:
            sleep(1) 

    finally:

        if not exit_value:
            inner()

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
        
    def send(self, _ ):

        if check_if_user_is_hf_user_and_has_hf_inference_credits():

            inference_client = InferenceClientModel(model_id=HUGGING_FACE_MODEL_ID, token=HfFolder.get_token())

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

    def __init__(self, agent_series : LanguageModelAgentGenerator = LanguageModelAgentGenerator(),  *args, **kwargs) -> None:    
        super().__init__(agent=next(agent_series), *args, **kwargs)
        self.agent_series = agent_series

    def refresh_agent(self) -> None:
        self.agent = next(self.agent_series)

    def launch(self, share : bool = True, **kwargs) -> None:
        """
        Create the Gradio app, mutate its event handlers directly (before launch),
        and ensure a pre_interact callable runs before the agent interaction.
        pre_interact signature: fn(*handler_args, **handler_kwargs) -> None
        """
        import gradio as gr

        app , pre_interact = self.create_app() , self.refresh_agent

        # If caller provided a pre_interact hook, try to attach it directly to the
        # app's input component(s) so it runs before the agent handler.

        print("Attaching refresh_agent hook to Gradio app input components...")

        try:
            # Find candidate input components (common names for text inputs)
            components = getattr(app, "components", None) or getattr(app, "children", []) or []
            inputs = [
                c for c in components
                if c.__class__.__name__ in ("Textbox", "TextArea", "TextInput")
            ]

            if not inputs:
                logging.info("No textbox-like component found to attach pre_interact hook.")
            else:
                # Attach to each found input: attempt to remove existing handlers then add our wrapped handler.
                for input_comp in inputs:
                    # Best-effort: try clearing internal handler lists so our handler becomes the primary.
                    # These internals may differ between Gradio versions; wrap in try/except.
                    try:
                        if hasattr(input_comp, "callbacks"):
                            # modern gradio may store callbacks here
                            input_comp.callbacks.clear()
                        if hasattr(input_comp, "_callbacks"):
                            input_comp._callbacks.clear()
                        if hasattr(input_comp, "events"):
                            input_comp.events.clear()
                    except Exception:
                        logging.debug("Could not clear existing callbacks for component %s", getattr(input_comp, "id", str(input_comp)))

                    # Build handler that runs pre_interact then delegates to the UI's interact_with_agent.
                    def _make_handler(comp):
                        def handler(*handler_args, **handler_kwargs):
                            try:
                                pre_interact(*handler_args, **handler_kwargs)
                            except Exception:
                                logging.error("pre_interact raised:\n" + format_exc())
                            # Call the UI's interact_with_agent if present; otherwise nothing.
                            try:
                                if hasattr(self, "interact_with_agent"):
                                    return self.interact_with_agent(*handler_args, **handler_kwargs)
                            except Exception:
                                logging.error("interact_with_agent raised after pre_interact:\n" + format_exc())
                            return None
                        return handler

                    wrapped = _make_handler(input_comp)

                    # Attach the wrapped handler. Prefer submit if available, else click.
                    try:
                        # Attempt to mirror a typical signature: inputs -> outputs unknown here.
                        # Registering with the component directly ensures the handler is invoked on submit/click.
                        if hasattr(input_comp, "submit"):
                            input_comp.submit(fn=wrapped, inputs=[input_comp], outputs=[input_comp])
                        elif hasattr(input_comp, "change"):
                            input_comp.change(fn=wrapped, inputs=[input_comp], outputs=[input_comp])
                        elif hasattr(input_comp, "click"):
                            input_comp.click(fn=wrapped, inputs=[input_comp], outputs=[input_comp])
                        else:
                            # Fallback: attach as a general event (Blocks API might accept .on_event)
                            try:
                                input_comp.on_event("submit", wrapped)
                            except Exception:
                                logging.debug("Could not attach handler using on_event for %s", getattr(input_comp, "id", str(input_comp)))
                    except Exception:
                        logging.error("Failed to attach wrapped handler to component %s:\n%s", getattr(input_comp, "id", str(input_comp)), format_exc())

        except Exception:
            logging.error("pre_interact attachment failed:\n" + format_exc())

        # Launch the (mutated) app
        app.launch(share=share, **kwargs)

def main() -> None:
    GradioUIWithBackupInference().launch(share=False, server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    main()