#!/usr/bin/env python3
"""Summarize GAP interface-ablation checkpoints, logs, and eval results."""
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional


EPOCH_RE = re.compile(r"Epoch\s+(\d+)\s+-\s+Avg Loss:\s+([0-9.]+)")


@dataclass(frozen=True)
class Variant:
    name: str
    ckpt_setting: str
    checkpoint_dir: str
    log_path: str
    question: str


def default_variants(task: str, task_config: str, expert_data_num: int, seed: int) -> list[Variant]:
    return [
        Variant(
            name="vanilla",
            ckpt_setting=task_config,
            checkpoint_dir=f"checkpoints/{task}_{task_config}_{expert_data_num}",
            log_path="",
            question="Full Pi3 scene tokens plus full future Pi3 latent target.",
        ),
        Variant(
            name="dino_only",
            ckpt_setting=f"{task_config}_dino_only_seed{seed}",
            checkpoint_dir=f"checkpoints/{task}_{task_config}_dino_only_seed{seed}_{expert_data_num}",
            log_path="logs/gap_ablate_dino_only_gpu1.log",
            question="No Pi3 observation and no future latent supervision.",
        ),
        Variant(
            name="no_future",
            ckpt_setting=f"{task_config}_no_future_seed{seed}",
            checkpoint_dir=f"checkpoints/{task}_{task_config}_no_future_seed{seed}_{expert_data_num}",
            log_path="logs/gap_ablate_no_future_gpu2.log",
            question="Full Pi3 observation, but future latent loss disabled.",
        ),
        Variant(
            name="pi3_pooled",
            ckpt_setting=f"{task_config}_pi3_pooled_seed{seed}",
            checkpoint_dir=f"checkpoints/{task}_{task_config}_pi3_pooled_seed{seed}_{expert_data_num}",
            log_path="logs/gap_ablate_pi3_pooled_gpu3.log",
            question="Pooled Pi3 tokens and pooled future target.",
        ),
        Variant(
            name="pi3_compressed",
            ckpt_setting=f"{task_config}_pi3_compressed_seed{seed}",
            checkpoint_dir=f"checkpoints/{task}_{task_config}_pi3_compressed_seed{seed}_{expert_data_num}",
            log_path="logs/gap_ablate_pi3_compressed_gpu1.log",
            question="Compressed Pi3 bottleneck and compressed future target.",
        ),
        Variant(
            name="pi3_random",
            ckpt_setting=f"{task_config}_pi3_random_seed{seed}",
            checkpoint_dir=f"checkpoints/{task}_{task_config}_pi3_random_seed{seed}_{expert_data_num}",
            log_path="logs/gap_ablate_pi3_random_gpu2.log",
            question="Random Pi3 token subset and random-token future target.",
        ),
        Variant(
            name="pi3_dropout",
            ckpt_setting=f"{task_config}_pi3_dropout_seed{seed}",
            checkpoint_dir=f"checkpoints/{task}_{task_config}_pi3_dropout_seed{seed}_{expert_data_num}",
            log_path="logs/gap_ablate_pi3_dropout_gpu3.log",
            question="Full future Pi3 target with observation token dropout.",
        ),
        Variant(
            name="pi3_eef_region",
            ckpt_setting=f"{task_config}_pi3_eef_region_seed{seed}",
            checkpoint_dir=f"checkpoints/{task}_{task_config}_pi3_eef_region_seed{seed}_{expert_data_num}",
            log_path="logs/gap_ablate_pi3_eef_region_gpu1.log",
            question="EEF-projected hand-near Pi3 token proxy and matching future target.",
        ),
        Variant(
            name="pi3_non_eef_region",
            ckpt_setting=f"{task_config}_pi3_non_eef_region_seed{seed}",
            checkpoint_dir=f"checkpoints/{task}_{task_config}_pi3_non_eef_region_seed{seed}_{expert_data_num}",
            log_path="logs/gap_ablate_pi3_non_eef_region_gpu3.log",
            question="Complement of the EEF-projected hand-near Pi3 proxy region.",
        ),
    ]


def latest_epoch_loss(log_file: Path) -> tuple[Optional[int], Optional[float]]:
    if not log_file.is_file():
        return None, None

    latest_epoch: Optional[int] = None
    latest_loss: Optional[float] = None
    with log_file.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            match = EPOCH_RE.search(line)
            if match:
                latest_epoch = int(match.group(1))
                latest_loss = float(match.group(2))
    return latest_epoch, latest_loss


def candidate_logs(root: Path, variant: Variant) -> list[Path]:
    if not variant.log_path:
        return []

    configured = root / variant.log_path
    matches = sorted((root / "logs").glob(f"gap_ablate_{variant.name}_gpu*.log"))
    if configured.is_file() and configured not in matches:
        matches.append(configured)
    return matches


def latest_epoch_loss_from_logs(log_files: Iterable[Path]) -> tuple[Optional[int], Optional[float]]:
    best_epoch: Optional[int] = None
    best_loss: Optional[float] = None
    for log_file in log_files:
        epoch, loss = latest_epoch_loss(log_file)
        if epoch is None:
            continue
        if best_epoch is None or epoch > best_epoch:
            best_epoch = epoch
            best_loss = loss
    return best_epoch, best_loss


def parse_success_rate(result_file: Path) -> Optional[float]:
    if not result_file.is_file():
        return None

    lines = [line.strip() for line in result_file.read_text(encoding="utf-8").splitlines()]
    for line in reversed(lines):
        if not line:
            continue
        try:
            return float(line)
        except ValueError:
            continue
    return None


def fmt_float(value: Optional[float], digits: int = 4) -> str:
    if value is None:
        return "pending"
    return f"{value:.{digits}f}"


def fmt_epoch_loss(epoch: Optional[int], loss: Optional[float]) -> str:
    if epoch is None:
        return "not found"
    return f"{epoch} / {fmt_float(loss)}"


def fmt_success_count(rate: Optional[float], test_num: int) -> str:
    if rate is None:
        return "pending"
    return f"{round(rate * test_num)}/{test_num}"


def fmt_delta(rate: Optional[float], baseline: Optional[float]) -> str:
    if rate is None or baseline is None:
        return "pending"
    return f"{rate - baseline:+.3f}"


def path_for_report(path: Path, root: Path) -> str:
    if not path.exists():
        return "pending"
    if path.is_relative_to(root):
        return str(path.relative_to(root))
    return str(path)


def saved_checkpoints(checkpoint_dir: Path) -> list[int]:
    if not checkpoint_dir.is_dir():
        return []
    epochs = []
    for path in checkpoint_dir.glob("*.ckpt"):
        try:
            epochs.append(int(path.stem))
        except ValueError:
            continue
    return sorted(epochs)


def checkpoint_status(checkpoint_dir: Path, target_checkpoint: int) -> str:
    epochs = saved_checkpoints(checkpoint_dir)
    if target_checkpoint in epochs:
        return f"{target_checkpoint} ready"
    if epochs:
        return f"{epochs[-1]} saved; {target_checkpoint} pending"
    return f"{target_checkpoint} pending"


def result_file_for(
    root: Path,
    results_dir: Path,
    task: str,
    policy: str,
    task_config: str,
    variant: Variant,
    seed: int,
    checkpoint: int,
) -> Path:
    return (
        results_dir
        / task
        / policy
        / task_config
        / variant.ckpt_setting
        / f"seed_{seed}"
        / str(checkpoint)
        / "_result.txt"
    )


def markdown_table(rows: Iterable[list[str]]) -> str:
    rows = list(rows)
    if not rows:
        return ""
    widths = [max(len(row[col]) for row in rows) for col in range(len(rows[0]))]
    out = []
    header = rows[0]
    out.append("| " + " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(header)) + " |")
    out.append("| " + " | ".join("-" * widths[i] for i in range(len(header))) + " |")
    for row in rows[1:]:
        out.append("| " + " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)) + " |")
    return "\n".join(out)


def build_report(args: argparse.Namespace) -> str:
    root = Path(args.root).resolve()
    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = root / results_dir
    variants = default_variants(args.task, args.task_config, args.expert_data_num, args.seed)

    rows = [[
        "variant",
        "checkpoint",
        "latest epoch/loss",
        "successes",
        "eval success",
        "delta vs vanilla",
        "result file",
        "question",
    ]]

    rates: dict[str, Optional[float]] = {}
    row_data = []
    for variant in variants:
        ckpt_dir = root / variant.checkpoint_dir
        result_path = result_file_for(
            root=root,
            results_dir=results_dir,
            task=args.task,
            policy=args.policy,
            task_config=args.task_config,
            variant=variant,
            seed=args.seed,
            checkpoint=args.checkpoint,
        )
        epoch, loss = latest_epoch_loss_from_logs(candidate_logs(root, variant))
        rate = parse_success_rate(result_path)
        rates[variant.name] = rate
        row_data.append((variant, ckpt_dir, result_path, epoch, loss, rate))

    baseline = rates.get("vanilla")
    for variant, ckpt_dir, result_path, epoch, loss, rate in row_data:
        rows.append([
            variant.name,
            checkpoint_status(ckpt_dir, args.checkpoint),
            fmt_epoch_loss(epoch, loss),
            fmt_success_count(rate, args.test_num),
            fmt_float(rate, digits=3),
            fmt_delta(rate, baseline),
            path_for_report(result_path, root),
            variant.question,
        ])

    completed_rates = {name: rate for name, rate in rates.items() if rate is not None}
    best_name = None
    best_rate = None
    if completed_rates:
        best_name, best_rate = max(completed_rates.items(), key=lambda item: item[1])

    lines = [
        "# GAP Interface Ablation Result Summary",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"Task: `{args.task}`  Config: `{args.task_config}`  Demos: `{args.expert_data_num}`  "
        f"Seed: `{args.seed}`  Checkpoint: `{args.checkpoint}`",
        "",
        f"Rollouts per variant: `{args.test_num}`",
        "",
        f"Results dir: `{results_dir.relative_to(root) if results_dir.is_relative_to(root) else results_dir}`",
        "",
        markdown_table(rows),
        "",
        "## Interpretation Guardrails",
        "",
        "- `dino_only` tests whether any Pi3/future-latent signal matters beyond DINOv3 plus proprioception.",
        "- `no_future` tests whether future Pi3 supervision matters when full Pi3 observation tokens remain.",
        "- `pi3_pooled`, `pi3_compressed`, `pi3_random`, and `pi3_dropout` test whether dense full-scene Pi3 token structure is necessary.",
        "- `pi3_eef_region` and `pi3_non_eef_region` use projected end-effector positions as a hand-near proxy/control. They are useful directional probes, not true object-hand masks.",
        "- Current zarr/HDF5 data does not include object masks, hand masks, object pose, contact labels, or non-empty point clouds. Therefore these runs cannot directly prove an object-hand-interaction-region mechanism; they can only show whether full-scene dense geometry appears necessary and whether an EEF-near proxy is competitive.",
        "",
        "## Current Readout",
        "",
    ]

    if baseline is None:
        lines.append("- Baseline eval is missing, so no comparison is available yet.")
    else:
        lines.append(
            f"- Vanilla baseline success rate is `{baseline:.3f}` "
            f"({fmt_success_count(baseline, args.test_num)})."
        )

    pending = [name for name, rate in rates.items() if rate is None and name != "vanilla"]
    if pending:
        lines.append(f"- Pending ablation evals: `{', '.join(pending)}`.")
    else:
        lines.append("- All configured ablation eval results are present.")
        if best_name is not None and best_rate is not None:
            lines.append(
                f"- Best observed variant is `{best_name}` at `{best_rate:.3f}` "
                f"({fmt_success_count(best_rate, args.test_num)})."
            )
        dino = rates.get("dino_only")
        no_future = rates.get("no_future")
        eef = rates.get("pi3_eef_region")
        non_eef = rates.get("pi3_non_eef_region")
        pooled = rates.get("pi3_pooled")
        if dino is not None and baseline is not None:
            if dino >= baseline:
                lines.append(
                    "- `dino_only` being at or above vanilla means this run does not support a claim "
                    "that full-scene Pi3 observation plus future latent is the dominant source of success."
                )
            else:
                lines.append(
                    "- `dino_only` below vanilla is consistent with Pi3/future-latent helping, "
                    "but by itself does not identify whether the useful signal is full-scene geometry."
                )
        if no_future is not None and baseline is not None:
            if no_future < baseline:
                lines.append(
                    "- `no_future` below vanilla suggests future-latent supervision may matter within the Pi3 setup, "
                    "but this is not enough to prove a full-scene geometry mechanism."
                )
            else:
                lines.append(
                    "- `no_future` at or above vanilla weakens the case that future-latent supervision is essential "
                    "in this run."
                )
        if eef is not None and non_eef is not None:
            if abs(eef - non_eef) <= (1.0 / args.test_num):
                lines.append(
                    "- `pi3_eef_region` and `pi3_non_eef_region` are close, so the EEF-near proxy is only a weak "
                    "directional signal in this single-task run."
                )
            elif eef > non_eef:
                lines.append(
                    "- `pi3_eef_region` above `pi3_non_eef_region` is directionally consistent with an interaction-region "
                    "hypothesis, but it remains an EEF proxy rather than a true object-hand mask."
                )
            else:
                lines.append(
                    "- `pi3_non_eef_region` above `pi3_eef_region` does not support the EEF-near proxy as the main useful region."
                )
        if pooled is not None and baseline is not None:
            if pooled >= baseline - (1.0 / args.test_num):
                lines.append(
                    "- `pi3_pooled` remaining competitive weakens a strict dense full-scene token requirement."
                )
            else:
                lines.append(
                    "- `pi3_pooled` below vanilla is consistent with some loss from compressing Pi3 token structure."
                )
        elif pooled is not None:
            lines.append(
                "- `pi3_pooled` should be compared against vanilla before making dense-token claims."
            )
        lines.append(
            "- This is still one task, one seed, and 50 demos; it is useful evidence for experiment direction, "
            "not a paper-scale conclusion."
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="GAP repository root")
    parser.add_argument("--task", default="place_dual_shoes")
    parser.add_argument("--task-config", default="demo_clean")
    parser.add_argument("--expert-data-num", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", type=int, default=200)
    parser.add_argument("--policy", default="GAP")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--test-num", type=int, default=10)
    parser.add_argument("--output", default="reports/gap_ablation_result_summary.md")
    args = parser.parse_args()

    report = build_report(args)
    if args.output == "-":
        print(report, end="")
        return

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = Path(args.root) / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
