#!/usr/bin/env python
"""Summarize the flow/action-UV regression gate results."""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any


DEFAULT_BASELINE_JSON = "reports/dual_shoes_eval100_existing_baselines.json"
DEFAULT_SIX_WAY_MD = "reports/flow_interaction_eval100_summary.md"
DEFAULT_RESULTS_ROOT = "results_flow_interaction_regression_gate"
DEFAULT_REPORT_MD = "reports/flow_interaction_regression_gate_report.md"
DEFAULT_REPORT_JSON = "reports/flow_interaction_regression_gate_results.json"
DEFAULT_ZARR = "data/place_dual_shoes-demo_clean-50-pi3-20-5.zarr"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the flow/action-UV regression-gate report.")
    parser.add_argument("--baseline-json", default=DEFAULT_BASELINE_JSON)
    parser.add_argument("--six-way-md", default=DEFAULT_SIX_WAY_MD)
    parser.add_argument("--results-root", default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--zarr", default=DEFAULT_ZARR)
    parser.add_argument("--out-md", default=DEFAULT_REPORT_MD)
    parser.add_argument("--out-json", default=DEFAULT_REPORT_JSON)
    parser.add_argument("--test-num", type=int, default=100)
    return parser.parse_args()


def load_existing_baselines(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows", payload if isinstance(payload, list) else [])
    keep = {"dino_only", "vanilla", "pi3_eef_region", "pi3_pooled"}
    return [row for row in rows if row.get("variant") in keep]


def parse_six_way_summary(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    in_table = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("| variant | eval100 success | count |"):
            in_table = True
            continue
        if not in_table:
            continue
        if not line.strip().startswith("|"):
            if rows:
                break
            continue
        if set(line.strip().replace("|", "").replace(" ", "")) <= {"-", ":"}:
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 3:
            continue
        count_match = re.search(r"(\d+)\s*/\s*(\d+)", cells[2])
        rows.append(
            {
                "variant": cells[0],
                "success_rate": float(cells[1]) if _is_float(cells[1]) else None,
                "success_count": int(count_match.group(1)) if count_match else None,
                "eval_rollouts": int(count_match.group(2)) if count_match else None,
            }
        )
    return rows


def _is_float(value: str) -> bool:
    try:
        float(value)
        return True
    except Exception:
        return False


def parse_result_file(path: Path, results_root: Path, test_num: int) -> dict[str, Any] | None:
    values = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if _is_float(stripped):
            values.append(float(stripped))
    if not values:
        return None
    try:
        stage = path.relative_to(results_root).parts[0]
    except Exception:
        stage = path.parent.name
    success_rate = values[-1]
    return {
        "stage": stage,
        "result_path": str(path),
        "success_rate": success_rate,
        "success_count": int(round(success_rate * test_num)),
        "eval_rollouts": test_num,
    }


def parse_gate_results(results_root: Path, test_num: int) -> list[dict[str, Any]]:
    if not results_root.exists():
        return []
    rows = []
    for path in sorted(results_root.rglob("_result.txt")):
        row = parse_result_file(path, results_root, test_num)
        if row is not None:
            rows.append(row)
    return rows


def total_zarr_steps(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        import zarr

        group = zarr.open(str(path), mode="r")
        data = group["data"] if "data" in group else group
        if "action" in data:
            return int(data["action"].shape[0])
    except Exception:
        return None
    return None


def estimate_updates(total_steps: int | None, epochs: int = 200) -> dict[str, Any]:
    if total_steps is None:
        total_steps = 11510
        source = "fallback"
    else:
        source = "zarr"
    batch32_per_epoch = math.ceil(total_steps / 32)
    batch256_per_epoch = math.ceil(total_steps / 256)
    return {
        "source": source,
        "total_steps": total_steps,
        "epochs": epochs,
        "batch32_updates_per_epoch": batch32_per_epoch,
        "batch32_total_updates": batch32_per_epoch * epochs,
        "batch256_updates_per_epoch": batch256_per_epoch,
        "batch256_total_updates": batch256_per_epoch * epochs,
        "batch32_vs_batch256_ratio": (batch32_per_epoch * epochs) / (batch256_per_epoch * epochs),
    }


def gate_by_stage(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for row in rows:
        stage = row.get("stage")
        if not stage:
            continue
        best[stage] = row
    return best


def diagnosis(gate_rows: list[dict[str, Any]]) -> list[str]:
    by_stage = gate_by_stage(gate_rows)
    lines = []
    old_auto = by_stage.get("re_eval_old_dino_auto")
    if old_auto is None:
        lines.append("Old DINO-only checkpoint re-eval is pending, so deploy/eval regression is not resolved yet.")
    elif old_auto["success_count"] >= 18:
        lines.append(
            f"Old DINO-only re-eval is close to the 24/100 baseline ({old_auto['success_count']}/100), so deploy/eval code is probably okay."
        )
    elif old_auto["success_count"] <= 6:
        lines.append(
            f"Old DINO-only re-eval dropped near the six-way DINO result ({old_auto['success_count']}/100), so deploy/eval regression is likely."
        )
    else:
        lines.append(
            f"Old DINO-only re-eval is intermediate ({old_auto['success_count']}/100); inspect EMA/model choice, env drift, and eval config."
        )

    repro = by_stage.get("train_repro_dino_auto")
    if repro is None:
        lines.append("Batch32 DINO-only reproduction is pending, so the batch256 training recipe remains an open confound.")
    elif repro["success_count"] >= 18:
        lines.append(
            f"Batch32 DINO-only reproduction recovered ({repro['success_count']}/100), so the batch256 ablation is not a fair method result."
        )
    elif repro["success_count"] <= 6:
        lines.append(
            f"Batch32 DINO-only reproduction still failed ({repro['success_count']}/100), so a training/code regression remains."
        )
    else:
        lines.append(
            f"Batch32 DINO-only reproduction is intermediate ({repro['success_count']}/100); treat conclusions as unresolved."
        )

    flow_rows = [row for row in gate_rows if row.get("stage", "").startswith("fair_flow")]
    if not flow_rows:
        lines.append("Fair flow runs are pending; do not interpret flow/action-UV negative results until DINO-only reproduction passes.")
    else:
        best_flow = max(flow_rows, key=lambda row: row["success_count"])
        lines.append(
            f"Best fair flow gate result so far is {best_flow['stage']} at {best_flow['success_count']}/100."
        )
    return lines


def decision_gate_status(gate_rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    by_stage = gate_by_stage(gate_rows)
    old_auto = by_stage.get("re_eval_old_dino_auto")
    repro = by_stage.get("train_repro_dino_auto")
    fair_flow_stage_names = {
        "fair_flow_dino_only_auto",
        "fair_flow_current_eef_auto",
        "fair_flow_action_uv_auto",
        "fair_flow_action_uv_expert_final_auto",
        "fair_flow_action_uv_flow_interp_auto",
        "fair_flow_action_uv_clean_only_auto",
    }
    fair_flow = [row for row in gate_rows if row.get("stage") in fair_flow_stage_names]

    if old_auto is None:
        q1 = "pending: re-eval old DINO-only checkpoint with current deploy/eval code."
    elif old_auto["success_count"] >= 18:
        q1 = f"answered yes: old DINO-only re-eval is close enough at {old_auto['success_count']}/100."
    elif old_auto["success_count"] <= 6:
        q1 = f"answered no: old DINO-only re-eval collapsed to {old_auto['success_count']}/100; deploy/eval regression likely."
    else:
        q1 = f"inconclusive: old DINO-only re-eval is {old_auto['success_count']}/100."

    if repro is None:
        q2 = "pending: train and eval a fresh batch32 DINO-only reproduction."
    elif repro["success_count"] >= 18:
        q2 = f"answered yes: batch32 DINO-only recovered to {repro['success_count']}/100; batch256 ablation is confounded."
    elif repro["success_count"] <= 6:
        q2 = f"answered no: batch32 DINO-only stayed low at {repro['success_count']}/100; training/code regression remains."
    else:
        q2 = f"inconclusive: batch32 DINO-only reached {repro['success_count']}/100."

    current_eef = by_stage.get("fair_flow_current_eef_auto")
    if not fair_flow:
        q3 = "pending: run fair batch32 flow gates after DINO-only reproduction passes."
    elif current_eef is None:
        best = max(fair_flow, key=lambda row: row["success_count"])
        q3 = (
            f"partial: best flow gate is {best['stage']} at {best['success_count']}/100, "
            "but fair_flow_current_eef_auto is still missing."
        )
    else:
        best = max(fair_flow, key=lambda row: row["success_count"])
        q3 = (
            f"answered with current fair flow evidence: best flow gate is {best['stage']} "
            f"at {best['success_count']}/100; current-EEF control is "
            f"{current_eef['success_count']}/100."
        )

    return [
        {"question": "Q1 old DINO-only current eval near 24/100", "status": q1},
        {"question": "Q2 fresh batch32 DINO-only reproduction", "status": q2},
        {"question": "Q3 fixed flow source under fair recipe", "status": q3},
    ]


def count_text(row: dict[str, Any]) -> str:
    count = row.get("success_count")
    total = row.get("eval_rollouts", 100)
    rate = row.get("success_rate")
    if count is None:
        return "pending"
    if rate is None:
        return f"{count}/{total}"
    return f"{count}/{total} ({rate:.2f})"


def write_report(
    out_md: Path,
    existing: list[dict[str, Any]],
    six_way: list[dict[str, Any]],
    gate_rows: list[dict[str, Any]],
    updates: dict[str, Any],
    diag_lines: list[str],
    gate_status: list[dict[str, str]],
) -> None:
    lines = [
        "# Flow Interaction Regression Gate Report",
        "",
        "## Existing Eval100 Baselines",
        "",
        "| variant | success | result |",
        "| --- | ---: | --- |",
    ]
    for row in existing:
        lines.append(f"| {row.get('variant')} | {count_text(row)} | `{row.get('result_path', '')}` |")
    if not existing:
        lines.append("| pending | pending | baseline JSON not found |")

    lines.extend(["", "## Latest Six-Way Eval100", "", "| variant | success |", "| --- | ---: |"])
    for row in six_way:
        lines.append(f"| {row.get('variant')} | {count_text(row)} |")
    if not six_way:
        lines.append("| pending | pending |")

    lines.extend(["", "## Decision Gate Status", "", "| question | status |", "| --- | --- |"])
    for row in gate_status:
        lines.append(f"| {row['question']} | {row['status']} |")

    lines.extend(
        [
            "",
            "## Milestone 3D Candidate-UV Context",
            "",
            "- `expert_final` supervises every intermediate/noised action candidate toward the final expert action-aligned UV, so it is not fully candidate-consistent for flow matching.",
            "- `flow_interp` uses the flow lambda to interpolate from current EEF UV to expert final UV, which is a better approximation of the candidate interaction field when true FK-derived candidate UV is unavailable.",
            "- `fair_flow_current_eef_auto` is required to separate gains from dynamic action-conditioned UV from gains that any EEF-local DINO token gives to flow.",
        ]
    )

    lines.extend(["", "## Regression Gate Results", "", "| stage | success | result |", "| --- | ---: | --- |"])
    for row in gate_rows:
        lines.append(f"| {row.get('stage')} | {count_text(row)} | `{row.get('result_path')}` |")
    if not gate_rows:
        lines.append("| pending | pending | no `_result.txt` files found |")

    lines.extend(
        [
            "",
            "## Optimizer Update Estimate",
            "",
            f"- zarr total steps: {updates['total_steps']} ({updates['source']})",
            f"- old/fair batch32: {updates['batch32_updates_per_epoch']} updates/epoch, {updates['batch32_total_updates']} updates over {updates['epochs']} epochs",
            f"- latest batch256: {updates['batch256_updates_per_epoch']} updates/epoch, {updates['batch256_total_updates']} updates over {updates['epochs']} epochs",
            f"- update ratio: batch32 is {updates['batch32_vs_batch256_ratio']:.1f}x batch256",
            "",
            "## Regression Diagnosis",
            "",
        ]
    )
    lines.extend(f"- {line}" for line in diag_lines)
    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            "Treat the batch256 six-way result as a recipe confound, not a method-level conclusion. Use the fair batch32 gate as the comparison point for further flow/action-UV analysis; after hold-source normalization, flow is non-collapsed but still below the best diffusion fair-recipe result in this seed.",
            "",
        ]
    )
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    existing = load_existing_baselines(Path(args.baseline_json))
    six_way = parse_six_way_summary(Path(args.six_way_md))
    gate_rows = parse_gate_results(Path(args.results_root), args.test_num)
    updates = estimate_updates(total_zarr_steps(Path(args.zarr)))
    diag_lines = diagnosis(gate_rows)
    gate_status = decision_gate_status(gate_rows)

    payload = {
        "existing_baselines": existing,
        "latest_six_way": six_way,
        "regression_gate": gate_rows,
        "optimizer_updates": updates,
        "diagnosis": diag_lines,
        "decision_gate_status": gate_status,
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    write_report(Path(args.out_md), existing, six_way, gate_rows, updates, diag_lines, gate_status)
    print(f"wrote: {args.out_md}")
    print(f"wrote: {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
