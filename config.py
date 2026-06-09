import os
import json
from pathlib import Path

def get_config_path():
    app_data = os.getenv("LOCALAPPDATA")
    if not app_data:
        app_data = os.path.expanduser("~")
    config_dir = Path(app_data) / "Parakeet"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir / "config.json"

def load_config():
    path = get_config_path()
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            pass
    return {"DEEPGRAM_API_KEY": "", "GROQ_API_KEY": ""}

def save_config(keys_dict):
    path = get_config_path()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(keys_dict, f)