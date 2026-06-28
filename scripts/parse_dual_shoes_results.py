#!/usr/bin/env python3
"""Parse public-50 Place Dual Shoes eval results into JSON and Markdown tables."""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class Variant:
    name: str
    ckpt_setting: str
    notes: str


OLD_VARIANTS = [
    Variant("vanilla", "demo_clean", "previous eval50 smoke evidence"),
    Variant("dino_only", "demo_clean_dino_only_seed0", "previous eval50 smoke evidence"),
    Variant("no_future", "demo_clean_no_future_seed0", "previous eval50 smoke evidence"),
    Variant("pi3_pooled", "demo_clean_pi3_pooled_seed0", "previous eval50 smoke evidence"),
    Variant("pi3_compressed", "demo_clean_pi3_compressed_seed0", "previous eval50 smoke evidence"),
    Variant("pi3_random", "demo_clean_pi3_random_seed0", "previous eval50 smoke evidence"),
    Variant("pi3_dropout", "demo_clean_pi3_dropout_seed0", "previous eval50 smoke evidence"),
    Variant("pi3_eef_region", "demo_clean_pi3_eef_region_seed0", "EEF proxy, not true object-hand"),
    Variant("pi3_non_eef_region", "demo_clean_pi3_non_eef_region_seed0", "EEF proxy complement"),
]

EVAL100_BASELINES = OLD_VARIANTS

TRAJECTORY_VARIANTS = [
    Variant("delta_future", "delta_future_seed0", "future_target_mode=pi3_delta"),
    Variant("changed_token_future", "changed_token_future_seed0", "future_target_mode=pi3_changed_tokens"),
    Variant("delta_changed_token_future", "delta_changed_token_future_seed0", "optional pi3_delta_changed_tokens"),
]


def parse_rate(path: Path) -> Optional[float]:
    if not path.is_file():
        return None
    for line in reversed(path.read_text(encoding="utf-8", errors="replace").splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            return float(line)
        except ValueError:
            continue
    return None


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 0.0
    phat = successes / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2.0 * n)) / denom
    radius = z * math.sqrt((phat * (1.0 - phat) + z * z / (4.0 * n)) / n) / denom
    return max(0.0, center - radius), min(1.0, center + radius)


def result_path(root: Path, task: str, task_config: str, variant: Variant, seed: int, checkpoint: int) -> Path:
    return root / task / "GAP" / task_config / variant.ckpt_setting / f"seed_{seed}" / str(checkpoint) / "_result.txt"


def path_for_report(path: Path, repo: Path) -> str:
    try:
        return str(path.relative_to(repo))
    except ValueError:
        return str(path)


def row_for(
    repo: Path,
    results_root: Path,
    task: str,
    task_config: str,
    demos: int,
    seed: int,
    checkpoint: int,
    rollouts: int,
    variant: Variant,
    group: str,
) -> dict:
    path = result_path(results_root, task, task_config, variant, seed, checkpoint)
    rate = parse_rate(path)
    successes = None if rate is None else int(round(rate * rollouts))
    if successes is None:
        ci_low = None
        ci_high = None
    else:
        ci_low, ci_high = wilson_interval(successes, rollouts)
    return {
        "group": group,
        "variant": variant.name,
        "training_demos": demos,
        "seed": seed,
        "checkpoint": checkpoint,
        "eval_rollouts": rollouts,
        "success_count": successes,
        "success_rate": rate,
        "wilson_95_ci": None if ci_low is None else [ci_low, ci_high],
        "result_path": path_for_report(path, repo),
        "result_exists": path.is_file(),
        "notes": variant.notes,
    }


def fmt_rate(value: Optional[float]) -> str:
    return "pending" if value is None else f"{value:.3f}"


def fmt_count(value: Optional[int], n: int) -> str:
    return "pending" if value is None else f"{value}/{n}"


def fmt_ci(value: Optional[list[float]]) -> str:
    if value is None:
        return "pending"
    return f"[{value[0]:.3f}, {value[1]:.3f}]"


def markdown_table(rows: list[dict]) -> str:
    headers = [
        "variant",
        "training demos",
        "seed",
        "checkpoint",
        "eval rollouts",
        "success count",
        "success rate",
        "Wilson 95% CI",
        "result path",
        "notes",
    ]
    table = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        table.append(
            "| "
            + " | ".join(
                [
                    str(row["variant"]),
                    str(row["training_demos"]),
                    str(row["seed"]),
                    str(row["checkpoint"]),
                    str(row["eval_rollouts"]),
                    fmt_count(row["success_count"], row["eval_rollouts"]),
                    fmt_rate(row["success_rate"]),
                    fmt_ci(row["wilson_95_ci"]),
                    row["result_path"],
                    row["notes"],
                ]
            )
            + " |"
        )
    return "\n".join(table)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--task", default="place_dual_shoes")
    parser.add_argument("--task-config", default="demo_clean")
    parser.add_argument("--expert-data-num", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", type=int, default=200)
    parser.add_argument("--eval50-root", default="results_eval50_full")
    parser.add_argument("--eval100-root", default="results_dual_shoes_50demo_eval100")
    parser.add_argument("--trajectory-root", default="results_dual_shoes_50demo_trajectory_eval100")
    parser.add_argument("--json-output", default="reports/dual_shoes_trajectory_results.json")
    parser.add_argument("--table-output", default="reports/dual_shoes_trajectory_results_table.md")
    parser.add_argument("--baseline-json-output", default="reports/dual_shoes_eval100_existing_baselines.json")
    args = parser.parse_args()

    repo = Path(args.root).resolve()
    eval50_root = (repo / args.eval50_root).resolve()
    eval100_root = (repo / args.eval100_root).resolve()
    trajectory_root = (repo / args.trajectory_root).resolve()

    rows: list[dict] = []
    for variant in OLD_VARIANTS:
        rows.append(
            row_for(
                repo,
                eval50_root,
                args.task,
                args.task_config,
                args.expert_data_num,
                args.seed,
                args.checkpoint,
                50,
                variant,
                "previous_eval50",
            )
        )
    for variant in EVAL100_BASELINES:
        rows.append(
            row_for(
                repo,
                eval100_root,
                args.task,
                args.task_config,
                args.expert_data_num,
                args.seed,
                args.checkpoint,
                100,
                variant,
                "existing_baseline_eval100",
            )
        )
    for variant in TRAJECTORY_VARIANTS:
        rows.append(
            row_for(
                repo,
                trajectory_root,
                args.task,
                args.task_config,
                args.expert_data_num,
                args.seed,
                args.checkpoint,
                100,
                variant,
                "trajectory_eval100",
            )
        )

    payload = {
        "task": args.task,
        "task_config": args.task_config,
        "expert_data_num": args.expert_data_num,
        "seed": args.seed,
        "checkpoint": args.checkpoint,
        "external_issue_note": "GitHub issue reports official released 50-demo vanilla around 11/100.",
        "caution": "Do not overclaim based on one seed or compare these public-50 results to the paper main table.",
        "rows": rows,
    }

    json_output = repo / args.json_output
    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    baseline_rows = [row for row in rows if row["group"] == "existing_baseline_eval100"]
    baseline_json_output = repo / args.baseline_json_output
    baseline_json_output.parent.mkdir(parents=True, exist_ok=True)
    baseline_json_output.write_text(json.dumps({"rows": baseline_rows}, indent=2), encoding="utf-8")

    table_output = repo / args.table_output
    table_output.parent.mkdir(parents=True, exist_ok=True)
    table_output.write_text(
        "# Dual Shoes Public-50 Trajectory Results\n\n"
        "GitHub issue note: official released 50-demo vanilla has been reported around 11/100.\n\n"
        "These rows are public-50 mechanism evidence, not paper-main reproduction evidence.\n\n"
        + markdown_table(rows)
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {json_output}")
    print(f"Wrote {baseline_json_output}")
    print(f"Wrote {table_output}")


if __name__ == "__main__":
    main()
