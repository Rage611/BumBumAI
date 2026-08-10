import json
import os
from pathlib import Path

CONFIG_PATH = Path(__file__).parent / "config.json"

def load_config() -> dict:
    if not CONFIG_PATH.exists():
        return {}
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {}

def save_config(keys: dict) -> None:
    # Ensure OPENAI_API_KEY and GROQ_API_KEY are saved.
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(keys, f, indent=4)