"""Load compact structured guidelines for LLM-based evaluation."""

from __future__ import annotations

import json
from pathlib import Path

GUIDELINES_PATH = Path(__file__).with_name("evaluation_guidelines.json")


def load_guidelines_json() -> str:
    with GUIDELINES_PATH.open("r", encoding="utf-8") as file:
        guidelines = json.load(file)
    return json.dumps(guidelines, ensure_ascii=False, separators=(",", ":"))
