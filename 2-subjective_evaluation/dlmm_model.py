"""Data-language mapping module for generating interaction prompts."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType

import pandas as pd

from src.subjective_eval_config import (
    DLMM_INDEX_CSV,
    PROMPT_CASES,
    PROMPT_DIR,
    PROMPTS_CSV,
    SAVE_INTERVAL,
    SRC_DIR,
    PromptCase,
)

INTEGER_COLUMNS = [
    "index",
    "scenario_idx",
    "start",
    "end",
    "lane_change_time_index",
    "lane_change_end_time_index",
]


def _read_index(target_id: int | None, prompt_cases: list[str] | None) -> pd.DataFrame:
    index_df = pd.read_csv(DLMM_INDEX_CSV)
    for column in INTEGER_COLUMNS:
        if column in index_df.columns:
            index_df[column] = pd.to_numeric(index_df[column], errors="coerce").astype("Int64")
    if target_id is not None:
        index_df = index_df[index_df["index"] == target_id]
    if prompt_cases is not None:
        unknown = sorted(set(prompt_cases) - set(PROMPT_CASES))
        if unknown:
            raise ValueError(f"Unknown prompt case(s): {unknown}")
        index_df = index_df[index_df["prompt_case"].isin(prompt_cases)]
    return index_df.reset_index(drop=True)


def _clear_generator_imports() -> None:
    for module_name in list(sys.modules):
        if module_name == "utils" or module_name.startswith("utils.") or module_name == "TwoDimTTC":
            del sys.modules[module_name]


def _load_prompt_module(prompt_case: PromptCase) -> ModuleType:
    script_path = SRC_DIR / prompt_case.script
    module_name = f"dlmm_generator_{prompt_case.name}"
    old_cwd = Path.cwd()
    old_sys_path = list(sys.path)
    try:
        os.chdir(SRC_DIR)
        if str(SRC_DIR) not in sys.path:
            sys.path.insert(0, str(SRC_DIR))
        spec = importlib.util.spec_from_file_location(module_name, script_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load prompt generator: {script_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        os.chdir(old_cwd)
        sys.path[:] = old_sys_path


def _normalize_prompt_result(result_df: pd.DataFrame, prompt_case: str) -> pd.DataFrame:
    if result_df.empty:
        return result_df
    result_df = result_df.copy()
    if "indexs" in result_df.columns and "index" not in result_df.columns:
        result_df = result_df.rename(columns={"indexs": "index"})
    if "prompt_case" not in result_df.columns:
        result_df.insert(0, "prompt_case", prompt_case)
    return result_df


def _save_outputs(output_frames: list[pd.DataFrame]) -> None:
    if not output_frames:
        return
    PROMPT_DIR.mkdir(parents=True, exist_ok=True)
    pd.concat(output_frames, ignore_index=True, sort=False).to_csv(PROMPTS_CSV, index=False, encoding="utf-8-sig")


def generate_prompts(
    target_id: int | None = None,
    prompt_cases: list[str] | None = None,
    save_interval: int = SAVE_INTERVAL,
) -> None:
    index_df = _read_index(target_id, prompt_cases)
    if index_df.empty:
        print("No matching interactions found.")
        return

    PROMPT_DIR.mkdir(parents=True, exist_ok=True)
    if PROMPTS_CSV.exists():
        PROMPTS_CSV.unlink()
    save_interval = max(1, save_interval)

    modules: dict[str, ModuleType] = {}
    output_frames: list[pd.DataFrame] = []
    generated_count = 0

    try:
        for row_number, (_, row) in enumerate(index_df.iterrows(), start=1):
            prompt_case_name = row["prompt_case"]
            prompt_case = PROMPT_CASES.get(prompt_case_name)
            if prompt_case is None:
                raise ValueError(f"Unknown prompt_case at row {row_number}: {prompt_case_name}")

            if prompt_case_name not in modules:
                modules[prompt_case_name] = _load_prompt_module(prompt_case)

            row_df = pd.DataFrame([row.to_dict()])
            result_df = modules[prompt_case_name].calculate_indicator(row_df)
            result_df = _normalize_prompt_result(result_df, prompt_case_name)
            if result_df.empty:
                continue

            output_frames.append(result_df)
            generated_count += len(result_df)
            if generated_count % save_interval == 0:
                _save_outputs(output_frames)
                print(f"Saved {generated_count} prompts to {PROMPTS_CSV}")
    except Exception:
        _save_outputs(output_frames)
        raise
    finally:
        _clear_generator_imports()

    _save_outputs(output_frames)
    print(f"Saved {generated_count} prompts to {PROMPTS_CSV}")


def run(
    target_id: int | None = None,
    prompt_cases: list[str] | None = None,
    save_interval: int = SAVE_INTERVAL,
) -> None:
    generate_prompts(target_id=target_id, prompt_cases=prompt_cases, save_interval=save_interval)


if __name__ == "__main__":
    run()
