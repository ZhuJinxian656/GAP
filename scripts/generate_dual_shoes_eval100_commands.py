#!/usr/bin/env python3
"""Generate eval100 commands for existing public-50 Place Dual Shoes checkpoints."""
from __future__ import annotations

import argparse
import os
import shlex
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Variant:
    name: str
    ckpt_setting: str
    checkpoint_dir: str
    note: str


def variants(task: str, task_config: str, demos: int, seed: int, include_optional: bool) -> list[Variant]:
    core = [
        Variant("vanilla", task_config, f"checkpoints/{task}_{task_config}_{demos}", "original GAP public-50 baseline"),
        Variant(
            "dino_only",
            f"{task_config}_dino_only_seed{seed}",
            f"checkpoints/{task}_{task_config}_dino_only_seed{seed}_{demos}",
            "DINO/state only, no Pi3 future loss",
        ),
        Variant(
            "no_future",
            f"{task_config}_no_future_seed{seed}",
            f"checkpoints/{task}_{task_config}_no_future_seed{seed}_{demos}",
            "Pi3 observations without future loss",
        ),
        Variant(
            "pi3_pooled",
            f"{task_config}_pi3_pooled_seed{seed}",
            f"checkpoints/{task}_{task_config}_pi3_pooled_seed{seed}_{demos}",
            "pooled Pi3 observation and future target",
        ),
    ]
    optional = [
        "pi3_eef_region",
        "pi3_non_eef_region",
        "pi3_dropout",
        "pi3_random",
        "pi3_compressed",
    ]
    if include_optional:
        core.extend(
            Variant(
                name,
                f"{task_config}_{name}_seed{seed}",
                f"checkpoints/{task}_{task_config}_{name}_seed{seed}_{demos}",
                "optional previous eval50 ablation checkpoint",
            )
            for name in optional
        )
    return core


def command_for(
    variant: Variant,
    task: str,
    task_config: str,
    demos: int,
    checkpoint: int,
    seed: int,
    gpu: str,
    test_num: int,
    results_root: str,
) -> str:
    prefix = f"RESULTS_ROOT={shlex.quote(results_root)}"
    args = [
        "bash",
        "eval.sh",
        task,
        task_config,
        variant.ckpt_setting,
        str(demos),
        str(checkpoint),
        str(gpu),
        str(seed),
        str(test_num),
    ]
    return prefix + " " + " ".join(shlex.quote(arg) for arg in args)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="place_dual_shoes")
    parser.add_argument("--task-config", default="demo_clean")
    parser.add_argument("--expert-data-num", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", type=int, default=200)
    parser.add_argument("--test-num", type=int, default=100)
    parser.add_argument("--gpu", default="1")
    parser.add_argument("--results-root", default="results_dual_shoes_50demo_eval100")
    parser.add_argument("--output", default="scripts/generated_dual_shoes_eval100_commands.sh")
    parser.add_argument("--include-optional", action="store_true")
    args = parser.parse_args()

    root = Path.cwd()
    results_root = args.results_root
    if not os.path.isabs(results_root):
        results_root = str(root / results_root)

    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Generated eval100 commands for public-50 Place Dual Shoes checkpoints.",
        "# Review GPU availability before running; commands write outside results_eval50_full.",
        "",
    ]
    written = 0
    skipped: list[str] = []
    for variant in variants(args.task, args.task_config, args.expert_data_num, args.seed, args.include_optional):
        ckpt_path = root / variant.checkpoint_dir / f"{args.checkpoint}.ckpt"
        if not ckpt_path.is_file():
            skipped.append(f"{variant.name}: missing {ckpt_path.relative_to(root)}")
            continue
        lines.append(f"# variant={variant.name} note={variant.note}")
        lines.append(
            command_for(
                variant=variant,
                task=args.task,
                task_config=args.task_config,
                demos=args.expert_data_num,
                checkpoint=args.checkpoint,
                seed=args.seed,
                gpu=args.gpu,
                test_num=args.test_num,
                results_root=results_root,
            )
        )
        lines.append("")
        written += 1

    if skipped:
        lines.append("# Skipped missing checkpoints:")
        lines.extend(f"# - {item}" for item in skipped)
        lines.append("")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    output.chmod(0o755)
    print(f"Wrote {written} eval commands to {output}")
    if skipped:
        print("Skipped:")
        for item in skipped:
            print(f"  {item}")


if __name__ == "__main__":
    main()
