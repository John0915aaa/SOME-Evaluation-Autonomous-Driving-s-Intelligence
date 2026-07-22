"""Shared paths and DLMM prompt-case settings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR.parent / "data"
OUTPUT_DIR = ROOT_DIR / "output"
SRC_DIR = ROOT_DIR / "src"
DLMM_INDEX_CSV = DATA_DIR / "dlmm_interaction_index.csv"
EVALUATION_SCORE_CSV = OUTPUT_DIR / "evaluation_score.csv"
SAVE_INTERVAL = 5


@dataclass(frozen=True)
class PromptCase:
    name: str
    script: str


PROMPT_CASES: dict[str, PromptCase] = {
    "mp1": PromptCase(
        name="mp1",
        script="get_prompt_mp1.py",
    ),
    "mp2": PromptCase(
        name="mp2",
        script="get_prompt_mp2.py",
    ),
    "cp1": PromptCase(
        name="cp1",
        script="get_prompt_cp1.py",
    ),
    "cp2": PromptCase(
        name="cp2",
        script="get_prompt_cp2.py",
    ),
    "f1": PromptCase(
        name="f1",
        script="get_prompt_f1.py",
    ),
    "f2": PromptCase(
        name="f2",
        script="get_prompt_f2.py",
    ),
}
