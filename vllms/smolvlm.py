import os
from pathlib import Path

from gradio_client import Client, handle_file

from vllms.base import BaseVLLM


class SmolVLM(BaseVLLM):
    def __init__(self):
        hf_token = os.environ.get("HUGGINGFACE_TOKEN", None)
        if not hf_token:
            raise ValueError("HUGGINGFACE_TOKEN env var not set")
        self.client = Client("HuggingFaceTB/SmolVLM", hf_token=hf_token)

    def prompt(self, instruction: str, image_path: Path) -> str:
        result = self.client.predict(
            input_dict={"text": instruction, "files": [handle_file(image_path)]},
            decoding_strategy="Greedy",
            temperature=0.4,
            max_new_tokens=512,
            repetition_penalty=1.2,
            top_p=0.8,
            api_name="/chat",
        )
        return result
