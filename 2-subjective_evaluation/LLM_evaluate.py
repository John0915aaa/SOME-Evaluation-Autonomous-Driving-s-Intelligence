"""Run GPT-4o-based subjective evaluation for DLMM prompts."""

from __future__ import annotations

import argparse

import pandas as pd

from src.evaluation_guidelines import load_guidelines_json
from src.llm_api_utils import GPT4O_MODEL, OpenAIClientPool, call_chat_completion
from src.subjective_eval_config import EVALUATION_RESULTS_CSV, PROMPTS_CSV, SAVE_INTERVAL

# This study used an online fine-tuned GPT-4o model. If fine-tuning is not needed,
# call the official GPT-4o model directly; if fine-tuning is needed, replace this
# model ID with your own fine-tuned model ID. OpenAI fine-tuning guide:
# https://platform.openai.com/docs/guides/fine-tuning
MODEL = GPT4O_MODEL

SYSTEM_PROMPT = "你是自动驾驶智能度评估专家，请基于自动驾驶车辆与其他车辆的交互信息评价自动驾驶智能度。"
GUIDELINES_JSON = load_guidelines_json()


def build_evaluation_input(prompt: str) -> str:
    return f"""请评价下面交互场景中自动驾驶车辆的智能度。

【交互测试内容】
{prompt}

【结构化评价准则 JSON】
{GUIDELINES_JSON}

请依据上述结构化准则进行综合判断，尤其优先考虑安全性，其次考虑舒适性、效率性、社会交互性和交通系统影响。回复为一段简洁中文分析，并在最后明确给出最终综合得分，格式必须为“最终综合得分：x/10”。
"""


def _read_prompts(target_id: int | None, prompt_cases: list[str] | None) -> pd.DataFrame:
    prompt_df = pd.read_csv(PROMPTS_CSV)
    if "indexs" in prompt_df.columns and "index" not in prompt_df.columns:
        prompt_df = prompt_df.rename(columns={"indexs": "index"})
    if target_id is not None:
        prompt_df = prompt_df[prompt_df["index"] == target_id]
    if prompt_cases is not None and "prompt_case" in prompt_df.columns:
        prompt_df = prompt_df[prompt_df["prompt_case"].isin(prompt_cases)]
    return prompt_df.reset_index(drop=True)


def _load_existing_results(overwrite: bool) -> tuple[list[dict], set[str]]:
    if overwrite or not EVALUATION_RESULTS_CSV.exists():
        return [], set()
    result_df = pd.read_csv(EVALUATION_RESULTS_CSV)
    if result_df.empty or "index" not in result_df.columns:
        return [], set()
    rows = result_df.to_dict("records")
    completed = {str(row["index"]) for row in rows}
    return rows, completed


def _save_results(rows: list[dict]) -> None:
    if not rows:
        return
    EVALUATION_RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(EVALUATION_RESULTS_CSV, index=False, encoding="utf-8-sig")


def evaluate_prompts(
    target_id: int | None = None,
    prompt_cases: list[str] | None = None,
    save_interval: int = SAVE_INTERVAL,
    overwrite: bool = False,
) -> None:
    prompt_df = _read_prompts(target_id, prompt_cases)
    if prompt_df.empty:
        print("No prompts found for evaluation.")
        return

    save_interval = max(1, save_interval)
    rows, completed_indexes = _load_existing_results(overwrite)
    client_pool = OpenAIClientPool()

    try:
        for row_number, (_, row) in enumerate(prompt_df.iterrows(), start=1):
            index = row["index"]
            if str(index) in completed_indexes:
                continue
            prompt = row["prompt"]
            prompt_case = row.get("prompt_case", "")
            user_input = build_evaluation_input(prompt)
            print(f"Evaluating index {index} ({row_number}/{len(prompt_df)})")
            response = call_chat_completion(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_input},
                ],
                client_pool=client_pool,
                model=MODEL,
                temperature=0.8,
                top_p=1.0,
            )
            rows.append({"index": index, "prompt_case": prompt_case, "response": response})
            completed_indexes.add(str(index))
            if len(rows) % save_interval == 0:
                _save_results(rows)
                print(f"Saved {len(rows)} evaluations to {EVALUATION_RESULTS_CSV}")
    except Exception:
        _save_results(rows)
        raise

    _save_results(rows)
    print(f"Saved {len(rows)} evaluations to {EVALUATION_RESULTS_CSV}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DLMM prompts with GPT-4o.")
    parser.add_argument("--target-id", type=int, default=None)
    parser.add_argument("--prompt-cases", nargs="*", default=None)
    parser.add_argument("--save-interval", type=int, default=SAVE_INTERVAL)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    evaluate_prompts(
        target_id=args.target_id,
        prompt_cases=args.prompt_cases,
        save_interval=args.save_interval,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
