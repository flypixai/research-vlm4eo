# VLLMs Geospatial Understanding

A research activity to quantify and evaluate the visual understanding of VLLMs in Geospatial imagery.

# Requirements

You need to have `uv` installed.

# How to run

1. Set your huggingface token:
```bash
HUGGINGFACE_TOKEN=hf_yourtoken
```

2. Run the LLMs:
```bash
uv run main.py
```

The models, prompts, and paths are configurable in the header of the `main.py` file

3. Assemble results into a CSV:
```bash
uv run assemble.py
```

This will produce a `results.csv` file.


