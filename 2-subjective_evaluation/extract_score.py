"""Extract numeric intelligence scores from LLM evaluation responses."""

from __future__ import annotations

import argparse
import re

import pandas as pd

from src.llm_api_utils import GPT4O_MODEL, OpenAIClientPool, call_chat_completion
from src.subjective_eval_config import EVALUATION_RESULTS_CSV, EVALUATION_SCORE_CSV, SAVE_INTERVAL

MODEL = GPT4O_MODEL
SYSTEM_PROMPT = "你是一名文本信息抽取专家，任务是从自动驾驶智能度评价文本中精确提取最终综合得分。"


def _regex_score(text: str) -> str | None:
    patterns = [
        r"最终综合得分[:：]\s*([0-9]+(?:\.[0-9]+)?)\s*/\s*10",
        r"综合得分[:：]\s*([0-9]+(?:\.[0-9]+)?)\s*/\s*10",
        r"得分[:：]\s*([0-9]+(?:\.[0-9]+)?)\s*/\s*10",
        r"([0-9]+(?:\.[0-9]+)?)\s*/\s*10",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            value = float(match.group(1))
            return f"{value:.1f}"
    return None


def extract_score_with_gpt(text: str, client_pool: OpenAIClientPool) -> str:
    user_prompt = f"""
从下面的自动驾驶智能度评价文本中提取最终综合得分。

要求：
1. 只返回一个保留一位小数的浮点数。
2. 不要返回任何解释或额外文字。
3. 如果文本中分数形式是“x/10”，直接取 x 作为结果。
4. 如果文本中存在多个分数，提取最终综合得分。

【文本】
{text}
"""
    result = call_chat_completion(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        client_pool=client_pool,
        model=MODEL,
        temperature=0.0,
        top_p=1.0,
        max_tokens=50,
    )
    match = re.search(r"[0-9]+(?:\.[0-9]+)?", result)
    if not match:
        raise ValueError(f"No numeric score found in LLM extraction result: {result}")
    return f"{float(match.group(0)):.1f}"


def _load_existing_scores(overwrite: bool) -> tuple[list[dict], set[str]]:
    if overwrite or not EVALUATION_SCORE_CSV.exists():
        return [], set()
    score_df = pd.read_csv(EVALUATION_SCORE_CSV)
    if score_df.empty or "index" not in score_df.columns:
        return [], set()
    rows = score_df.to_dict("records")
    completed = {str(row["index"]) for row in rows}
    return rows, completed


def _save_scores(rows: list[dict]) -> None:
    if not rows:
        return
    EVALUATION_SCORE_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(EVALUATION_SCORE_CSV, index=False, encoding="utf-8-sig")


def extract_scores(
    target_id: int | None = None,
    save_interval: int = SAVE_INTERVAL,
    overwrite: bool = False,
    regex_first: bool = True,
) -> None:
    result_df = pd.read_csv(EVALUATION_RESULTS_CSV)
    if target_id is not None:
        result_df = result_df[result_df["index"] == target_id]
    if result_df.empty:
        print("No evaluation responses found for score extraction.")
        return

    save_interval = max(1, save_interval)
    rows, completed_indexes = _load_existing_scores(overwrite)
    client_pool = None if regex_first else OpenAIClientPool()

    try:
        for row_number, (_, row) in enumerate(result_df.iterrows(), start=1):
            index = row["index"]
            if str(index) in completed_indexes:
                continue
            response_text = row["response"]
            score = _regex_score(response_text) if regex_first else None
            if score is None:
                if client_pool is None:
                    client_pool = OpenAIClientPool()
                score = extract_score_with_gpt(response_text, client_pool)
            output_row = {"index": index, "score": score}
            if "prompt_case" in row:
                output_row["prompt_case"] = row["prompt_case"]
            rows.append(output_row)
            completed_indexes.add(str(index))
            print(f"Extracted score for index {index}: {score} ({row_number}/{len(result_df)})")
            if len(rows) % save_interval == 0:
                _save_scores(rows)
                print(f"Saved {len(rows)} scores to {EVALUATION_SCORE_CSV}")
    except Exception:
        _save_scores(rows)
        raise

    _save_scores(rows)
    print(f"Saved {len(rows)} scores to {EVALUATION_SCORE_CSV}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract scores from LLM evaluation responses.")
    parser.add_argument("--target-id", type=int, default=None)
    parser.add_argument("--save-interval", type=int, default=SAVE_INTERVAL)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-regex-first", action="store_true")
    args = parser.parse_args()
    extract_scores(
        target_id=args.target_id,
        save_interval=args.save_interval,
        overwrite=args.overwrite,
        regex_first=not args.no_regex_first,
    )


if __name__ == "__main__":
    main()
