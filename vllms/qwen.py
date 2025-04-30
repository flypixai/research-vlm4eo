import os
from pathlib import Path

from gradio_client import Client, handle_file

from vllms.base import BaseVLLM


class Qwen2_5_7B(BaseVLLM):
    def __init__(self):
        hf_token = os.environ.get("HUGGINGFACE_TOKEN", None)
        if not hf_token:
            raise ValueError("HUGGINGFACE_TOKEN env var not set")
        self.client = Client("prithivMLmods/Qwen2.5-VL-7B-Instruct", hf_token=hf_token)

    def prompt(self, instruction: str, image_path: Path) -> str:
        result = self.client.predict(
            message={"text": instruction, "files": [handle_file(image_path)]},
            api_name="/chat",
        )
        return result
