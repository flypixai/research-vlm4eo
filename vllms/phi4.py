import os
from pathlib import Path

from gradio_client import Client, handle_file

from vllms.base import BaseVLLM


class Phi4(BaseVLLM):
    def __init__(self):
        hf_token = os.environ.get("HUGGINGFACE_TOKEN", None)
        if not hf_token:
            raise ValueError("HUGGINGFACE_TOKEN env var not set")
        self.client = Client("VIDraft/PHI4-Multimodal", hf_token=hf_token)

    def prompt(self, instruction: str, image_path: Path) -> str:
        result = self.client.predict(
            message={"text": instruction, "files": [handle_file(image_path)]},
            param_2=1024,
            param_3=0.6,
            param_4=0.9,
            param_5=50,
            param_6=1.2,
            api_name="/chat",
        )
        return result
