"""Extract and normalize all objective metrics with one context load per interaction."""

from __future__ import annotations

import argparse

import pandas as pd

from comfort_metrics import COMFORT_OUTPUT_FILES, calculate_comfort_for_context
from efficiency_metrics import calculate_avg_delay_for_context, calculate_task_time_for_context
from impact_metrics import calculate_impact_for_context
from interaction_metrics import calculate_interaction_for_context
from safety_metrics import calculate_pet_for_context, calculate_ttc_for_context
from src.objective_metrics_config import OUTPUT_DIR
from src.waymo_metric_utils import iter_contexts, write_csv

SAVE_INTERVAL = 30

COMBINED_COLUMNS = [
    "index",
    "TTC",
    "PET",
    "a_p",
    "a_l",
    "jerk",
    "yaw_rate",
    "task_time",
    "avg_delay",
    "IO",
    "impact",
]


def _write_outputs(
    output_rows: dict[str, list[dict]],
    combined_rows: list[dict],
) -> None:
    """Save all currently completed metric rows."""
    write_csv(output_rows["TTC"], OUTPUT_DIR / "safety" / "ttc_score.csv")
    write_csv(output_rows["PET"], OUTPUT_DIR / "safety" / "pet_score.csv")

    for metric_name, output_filename in COMFORT_OUTPUT_FILES.items():
        write_csv(
            output_rows[metric_name],
            OUTPUT_DIR / "comfort" / output_filename,
        )

    write_csv(
        output_rows["task_time"],
        OUTPUT_DIR / "efficiency" / "task_time.csv",
    )
    write_csv(
        output_rows["avg_delay"],
        OUTPUT_DIR / "efficiency" / "avg_delay.csv",
    )
    write_csv(
        output_rows["IO"],
        OUTPUT_DIR / "interaction" / "io.csv",
    )
    write_csv(
        output_rows["impact"],
        OUTPUT_DIR / "impact" / "impact.csv",
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    combined_df = pd.DataFrame(combined_rows, columns=COMBINED_COLUMNS)
    output_path = OUTPUT_DIR / "metrics.csv"
    combined_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Saved {len(combined_rows)} completed interaction(s) to {output_path}")


def run(target_id: int | None = None) -> None:
    output_rows = {
        "TTC": [],
        "PET": [],
        "a_p": [],
        "a_l": [],
        "jerk": [],
        "yaw_rate": [],
        "task_time": [],
        "avg_delay": [],
        "IO": [],
        "impact": [],
    }

    combined_rows: list[dict] = []
    seen_indexes: set[int] = set()

    try:
        for context in iter_contexts(target_id):
            if context.index in seen_indexes:
                raise ValueError(
                    f"Duplicate interaction index: {context.index}"
                )

            seen_indexes.add(context.index)
            print(
                f"Calculating all metrics for "
                f"interaction index={context.index}"
            )

            # Each context is loaded once and shared by all metric calculations.
            ttc_row = calculate_ttc_for_context(context)
            pet_row = calculate_pet_for_context(context)
            comfort_rows = calculate_comfort_for_context(context)
            task_time_row = calculate_task_time_for_context(context)
            avg_delay_row = calculate_avg_delay_for_context(context)
            io_row = calculate_interaction_for_context(context)
            impact_row = calculate_impact_for_context(context)

            output_rows["TTC"].append(ttc_row)
            output_rows["PET"].append(pet_row)

            for metric_name, row in comfort_rows.items():
                output_rows[metric_name].append(row)

            output_rows["task_time"].append(task_time_row)
            output_rows["avg_delay"].append(avg_delay_row)
            output_rows["IO"].append(io_row)
            output_rows["impact"].append(impact_row)

            combined_rows.append({
                "index": context.index,
                "TTC": ttc_row["TTC"],
                "PET": pet_row["PET"],
                "a_p": comfort_rows["a_p"]["a_p"],
                "a_l": comfort_rows["a_l"]["a_l"],
                "jerk": comfort_rows["jerk"]["jerk"],
                "yaw_rate": comfort_rows["yaw_rate"]["yaw_rate"],
                "task_time": task_time_row["task_time"],
                "avg_delay": avg_delay_row["avg_delay"],
                "IO": io_row["IO"],
                "impact": impact_row["impact"],
            })

            completed_count = len(combined_rows)

            if completed_count % SAVE_INTERVAL == 0:
                print(
                    f"Checkpoint reached: "
                    f"{completed_count} interactions completed."
                )
                _write_outputs(output_rows, combined_rows)

    except Exception:
        if combined_rows:
            print(
                "An error occurred. Saving all completed "
                "interactions before exiting."
            )
            _write_outputs(output_rows, combined_rows)
        raise

    # Save the final incomplete batch. If the total is exactly divisible by 30,
    # the latest checkpoint already contains all rows, so no repeated save is needed.
    if combined_rows and len(combined_rows) % SAVE_INTERVAL != 0:
        _write_outputs(output_rows, combined_rows)
    elif not combined_rows:
        _write_outputs(output_rows, combined_rows)

    print(f"Completed {len(combined_rows)} interaction(s).")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract and normalize all 10 objective metrics in one pass."
    )
    parser.add_argument("--target-id", type=int, default=None)
    args = parser.parse_args()
    run(target_id=args.target_id)


if __name__ == "__main__":
    main()