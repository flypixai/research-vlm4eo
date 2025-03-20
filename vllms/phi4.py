import os
from pathlib import Path

from gradio_client import Client, handle_file

from vllms.base import BaseVLLM


class Phi4(BaseVLLM):
    def __init__(self):
        hf_token = os.environ.get("HUGGINGFACE_TOKEN", None)
        if not hf_token:
            raise ValueError("HUGGINGFACE_TOKEN env var not set")
        self.client = Client("microsoft/phi-4-multimodal", hf_token=hf_token)

    def prompt(self, instruction: str, image_path: Path) -> str:
        self.client.predict(
            history=[],
            message={"text": instruction, "files": [handle_file(image_path)]},
            api_name="/process_message",
        )

        result = self.client.predict(
            history=[
                {
                    "role": "user",
                    "metadata": None,
                    "content": instruction,
                    "options": None,
                },
                {
                    "role": "user",
                    "metadata": None,
                    "content": {
                        "file": handle_file(image_path),
                        "alt_text": None,
                    },
                    "options": None,
                },
            ],
            api_name="/bot_response",
        )

        return result[-1]["content"]
