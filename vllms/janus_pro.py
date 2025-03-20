import os
from pathlib import Path

from gradio_client import Client, handle_file

from vllms.base import BaseVLLM


class JanusPro7B(BaseVLLM):
    def __init__(self):
        hf_token = os.environ.get("HUGGINGFACE_TOKEN", None)
        if not hf_token:
            raise ValueError("HUGGINGFACE_TOKEN env var not set")
        self.client = Client("deepseek-ai/Janus-Pro-7B", hf_token=hf_token)

    def prompt(self, instruction: str, image_path: Path) -> str:
        result = self.client.predict(
            image=handle_file(image_path),
            question=instruction,
            api_name="/multimodal_understanding",
        )
        return result
