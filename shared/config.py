"""Utility for loading YAML configs and .env files."""
from __future__ import annotations
import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# Load .env on first import
load_dotenv()


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file and return its contents as a dict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_api_key(var: str = "GROQ_API_KEY") -> str:
    """Return an API key from the environment; raise if missing."""
    key = os.environ.get(var, "").strip()
    if not key:
        raise EnvironmentError(
            f"Environment variable '{var}' is not set. "
            f"Copy .env.example to .env and fill in your key."
        )
    return key
