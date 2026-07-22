"""Generate a DLMM prompt, evaluate it, and extract its score in one pass."""

from __future__ import annotations

import argparse
import importlib.util
import os
import re
import sys
from pathlib import Path
from types import ModuleType

import pandas as pd

from src.evaluation_guidelines import load_guidelines_json
from src.llm_api_utils import GPT4O_MODEL, OpenAIClientPool, call_chat_completion
from src.subjective_eval_config import (
    DLMM_INDEX_CSV,
    EVALUATION_SCORE_CSV,
    PROMPT_CASES,
    SAVE_INTERVAL,
    SRC_DIR,
    PromptCase,
)

MODEL = GPT4O_MODEL
EVALUATION_SYSTEM_PROMPT = (
    "你是自动驾驶智能度评估专家，请基于自动驾驶车辆与其他车辆的交互信息评价自动驾驶智能度。"
)
EXTRACTION_SYSTEM_PROMPT = (
    "你是一名文本信息抽取专家，任务是从自动驾驶智能度评价文本中精确提取最终综合得分。"
)
GUIDELINES_JSON = load_guidelines_json()
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
    missing_columns = {"index", "prompt_case"} - set(index_df.columns)
    if missing_columns:
        raise ValueError(
            f"{DLMM_INDEX_CSV} is missing required columns: {sorted(missing_columns)}"
        )
    for column in INTEGER_COLUMNS:
        if column in index_df.columns:
            index_df[column] = pd.to_numeric(index_df[column], errors="coerce").astype("Int64")
    if index_df["index"].isna().any():
        raise ValueError(f"{DLMM_INDEX_CSV} contains an invalid or empty index.")
    if index_df["index"].duplicated().any():
        duplicate_indexes = index_df.loc[index_df["index"].duplicated(), "index"].tolist()
        raise ValueError(f"Duplicate interaction indexes: {duplicate_indexes[:10]}")

    if target_id is not None:
        index_df = index_df[index_df["index"] == target_id]
    if prompt_cases is not None:
        unknown_cases = sorted(set(prompt_cases) - set(PROMPT_CASES))
        if unknown_cases:
            raise ValueError(f"Unknown prompt case(s): {unknown_cases}")
        index_df = index_df[index_df["prompt_case"].isin(prompt_cases)]
    return index_df.reset_index(drop=True)


def _clear_generator_imports() -> None:
    for module_name in list(sys.modules):
        if module_name == "utils" or module_name.startswith("utils.") or module_name == "TwoDimTTC":
            del sys.modules[module_name]


def _load_prompt_module(prompt_case: PromptCase) -> ModuleType:
    script_path = SRC_DIR / prompt_case.script
    module_name = f"integrated_dlmm_generator_{prompt_case.name}"
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


def _generate_prompt(row: pd.Series, prompt_module: ModuleType) -> str:
    # Passing exactly one row guarantees that its trajectory context is read
    # once by the selected DLMM generator.
    result_df = prompt_module.calculate_indicator(pd.DataFrame([row.to_dict()]))
    if result_df.empty:
        raise RuntimeError(f"DLMM generated no prompt for index={int(row['index'])}.")
    if "indexs" in result_df.columns and "index" not in result_df.columns:
        result_df = result_df.rename(columns={"indexs": "index"})
    if "prompt" not in result_df.columns:
        raise ValueError(
            f"DLMM result for index={int(row['index'])} does not contain a prompt column."
        )
    if len(result_df) != 1:
        raise ValueError(
            f"DLMM generated {len(result_df)} prompts for one index={int(row['index'])}; expected 1."
        )
    prompt = result_df.iloc[0]["prompt"]
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"DLMM generated an empty prompt for index={int(row['index'])}.")
    return prompt.strip()


def build_evaluation_input(prompt: str) -> str:
    return f"""请评价下面交互场景中自动驾驶车辆的智能度。

【交互测试内容】
{prompt}

【结构化评价准则 JSON】
{GUIDELINES_JSON}

请依据上述结构化准则进行综合判断，尤其优先考虑安全性，其次考虑舒适性、效率性、社会交互性和交通系统影响。回复为一段简洁中文分析，并在最后明确给出最终综合得分，格式必须为“最终综合得分：x/10”。
"""


def _evaluate_prompt(prompt: str, client_pool: OpenAIClientPool, model: str) -> str:
    return call_chat_completion(
        [
            {"role": "system", "content": EVALUATION_SYSTEM_PROMPT},
            {"role": "user", "content": build_evaluation_input(prompt)},
        ],
        client_pool=client_pool,
        model=model,
        temperature=0.8,
        top_p=1.0,
    )


def _normalized_score(value: float) -> float:
    """Validate a score that is already normalized to the [0, 1] range."""
    score = float(value)
    if not 0.0 <= score <= 1.0:
        raise ValueError(f"Normalized score must be within [0, 1], got {score}.")
    return score


def _normalize_score_out_of_ten(value: float) -> float:
    """Convert an LLM score expressed as x/10 to the [0, 1] training scale."""
    score_out_of_ten = float(value)
    if not 0.0 <= score_out_of_ten <= 10.0:
        raise ValueError(
            f"Extracted score must be within [0, 10], got {score_out_of_ten}."
        )
    return round(score_out_of_ten / 10.0, 2)


def _regex_score(text: str) -> float | None:
    patterns = [
        r"最终综合得分[:：]\s*([0-9]+(?:\.[0-9]+)?)\s*/\s*10",
        r"综合得分[:：]\s*([0-9]+(?:\.[0-9]+)?)\s*/\s*10",
        r"得分[:：]\s*([0-9]+(?:\.[0-9]+)?)\s*/\s*10",
        r"([0-9]+(?:\.[0-9]+)?)\s*/\s*10",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return _normalize_score_out_of_ten(float(match.group(1)))
    return None


def _extract_score_with_gpt(
    evaluation_text: str,
    client_pool: OpenAIClientPool,
    model: str,
) -> float:
    extraction_input = f"""从下面的自动驾驶智能度评价文本中提取最终综合得分。

要求：
1. 只返回一个保留一位小数的浮点数。
2. 不要返回任何解释或额外文字。
3. 如果文本中分数形式是“x/10”，直接取 x 作为结果。
4. 如果文本中存在多个分数，提取最终综合得分。

【文本】
{evaluation_text}
"""
    result = call_chat_completion(
        [
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": extraction_input},
        ],
        client_pool=client_pool,
        model=model,
        temperature=0.0,
        top_p=1.0,
        max_tokens=50,
    )
    match = re.search(r"[0-9]+(?:\.[0-9]+)?", result)
    if not match:
        raise ValueError(f"No numeric score found in LLM extraction result: {result}")
    return _normalize_score_out_of_ten(float(match.group(0)))

def _extract_score(
    evaluation_text: str,
    client_pool: OpenAIClientPool,
    model: str,
    regex_first: bool,
) -> float:
    score = _regex_score(evaluation_text) if regex_first else None
    if score is not None:
        return score
    return _extract_score_with_gpt(evaluation_text, client_pool, model)


def _index_key(index: object) -> str:
    numeric_index = float(index)
    return str(int(numeric_index)) if numeric_index.is_integer() else str(numeric_index)


def _load_existing_scores(overwrite: bool) -> tuple[list[dict], set[str]]:
    if overwrite or not EVALUATION_SCORE_CSV.exists():
        return [], set()
    score_df = pd.read_csv(EVALUATION_SCORE_CSV)
    missing_columns = {"index", "score"} - set(score_df.columns)
    if missing_columns:
        raise ValueError(
            f"{EVALUATION_SCORE_CSV} is missing required columns: {sorted(missing_columns)}"
        )
    score_df = score_df[["index", "score"]].copy()
    score_df["score"] = pd.to_numeric(score_df["score"], errors="raise")
    score_df = score_df.drop_duplicates(subset=["index"], keep="last")
    rows = [
        {"index": int(float(row["index"])), "score": _normalized_score(row["score"])}
        for _, row in score_df.iterrows()
    ]
    return rows, {_index_key(row["index"]) for row in rows}


def _save_scores(rows: list[dict]) -> None:
    EVALUATION_SCORE_CSV.parent.mkdir(parents=True, exist_ok=True)
    score_df = pd.DataFrame(rows, columns=["index", "score"])
    score_df.to_csv(EVALUATION_SCORE_CSV, index=False, encoding="utf-8-sig")


def run(
    target_id: int | None = None,
    prompt_cases: list[str] | None = None,
    save_interval: int = SAVE_INTERVAL,
    overwrite: bool = False,
    regex_first: bool = True,
    model: str = MODEL,
) -> None:
    index_df = _read_index(target_id, prompt_cases)
    if index_df.empty:
        print("No matching interactions found.")
        return

    save_interval = max(1, save_interval)
    rows, completed_indexes = _load_existing_scores(overwrite)
    prompt_modules: dict[str, ModuleType] = {}
    client_pool: OpenAIClientPool | None = None
    newly_completed = 0

    try:
        for row_number, (_, row) in enumerate(index_df.iterrows(), start=1):
            index = int(row["index"])
            if _index_key(index) in completed_indexes:
                continue

            prompt_case_name = str(row["prompt_case"])
            prompt_case = PROMPT_CASES.get(prompt_case_name)
            if prompt_case is None:
                raise ValueError(
                    f"Unknown prompt_case={prompt_case_name!r} for index={index}."
                )
            if prompt_case_name not in prompt_modules:
                prompt_modules[prompt_case_name] = _load_prompt_module(prompt_case)

            print(
                f"Processing index={index}: prompt -> evaluation -> score "
                f"({row_number}/{len(index_df)})"
            )
            prompt = _generate_prompt(row, prompt_modules[prompt_case_name])
            if client_pool is None:
                client_pool = OpenAIClientPool()
            evaluation_text = _evaluate_prompt(prompt, client_pool, model)
            score = _extract_score(evaluation_text, client_pool, model, regex_first)

            rows.append({"index": index, "score": score})
            completed_indexes.add(_index_key(index))
            newly_completed += 1

            if newly_completed % save_interval == 0:
                _save_scores(rows)
                print(f"Saved {len(rows)} scores to {EVALUATION_SCORE_CSV}")
    except (Exception, KeyboardInterrupt):
        _save_scores(rows)
        raise
    finally:
        _clear_generator_imports()

    _save_scores(rows)
    print(f"Saved {len(rows)} scores to {EVALUATION_SCORE_CSV}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate DLMM prompts, evaluate them, and extract scores in one pass."
    )
    parser.add_argument("--target-id", type=int, default=None)
    parser.add_argument("--prompt-cases", nargs="*", default=None)
    parser.add_argument("--save-interval", type=int, default=SAVE_INTERVAL)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-regex-first", action="store_true")
    parser.add_argument("--model", default=MODEL)
    args = parser.parse_args()
    run(
        target_id=args.target_id,
        prompt_cases=args.prompt_cases,
        save_interval=args.save_interval,
        overwrite=args.overwrite,
        regex_first=not args.no_regex_first,
        model=args.model,
    )


if __name__ == "__main__":
    main()
