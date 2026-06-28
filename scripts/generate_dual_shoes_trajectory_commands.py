#!/usr/bin/env python3
"""Generate train/eval commands for trajectory-coupled future Pi3 variants."""
from __future__ import annotations

import argparse
import os
import shlex
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Variant:
    name: str
    checkpoint_tag: str
    overrides: list[str]


def variant_specs(include_delta_changed: bool) -> list[Variant]:
    specs = [
        Variant(
            name="delta_future",
            checkpoint_tag="delta_future_seed0",
            overrides=[
                "latent_mode=pi3_full",
                "use_future_loss=true",
                "future_target_mode=pi3_delta",
                "use_pi3_features=true",
                "policy.use_pi3_features=true",
            ],
        ),
        Variant(
            name="changed_token_future",
            checkpoint_tag="changed_token_future_seed0",
            overrides=[
                "latent_mode=pi3_full",
                "use_future_loss=true",
                "future_target_mode=pi3_changed_tokens",
                "future.changed_token_percentile=90",
                "future.changed_token_stopgrad_mask=true",
                "use_pi3_features=true",
                "policy.use_pi3_features=true",
            ],
        ),
    ]
    if include_delta_changed:
        specs.append(
            Variant(
                name="delta_changed_token_future",
                checkpoint_tag="delta_changed_token_future_seed0",
                overrides=[
                    "latent_mode=pi3_full",
                    "use_future_loss=true",
                    "future_target_mode=pi3_delta_changed_tokens",
                    "future.changed_token_percentile=90",
                    "future.changed_token_stopgrad_mask=true",
                    "use_pi3_features=true",
                    "policy.use_pi3_features=true",
                ],
            )
        )
    return specs


def q(value: str) -> str:
    return shlex.quote(str(value))


def train_command(
    variant: Variant,
    task: str,
    task_config: str,
    demos: int,
    seed: int,
    gpu: str,
    batch_size: int,
    num_epochs: int,
    checkpoint_every: int,
) -> str:
    exp_name = f"{task}_{task_config}_{demos}_{variant.name}_seed{seed}"
    overrides = variant.overrides + [
        f"checkpoint_tag={variant.checkpoint_tag}",
        f"exp_name={exp_name}",
        f"logging.name={exp_name}",
    ]
    args = [
        "bash",
        "train.sh",
        task,
        task_config,
        str(demos),
        str(seed),
        str(gpu),
        str(batch_size),
        str(num_epochs),
        str(checkpoint_every),
        *overrides,
    ]
    return " ".join(q(arg) for arg in args)


def eval_command(
    variant: Variant,
    task: str,
    task_config: str,
    demos: int,
    seed: int,
    gpu: str,
    checkpoint: int,
    test_num: int,
    results_root: str,
) -> str:
    args = [
        "bash",
        "eval.sh",
        task,
        task_config,
        variant.checkpoint_tag,
        str(demos),
        str(checkpoint),
        str(gpu),
        str(seed),
        str(test_num),
    ]
    return f"RESULTS_ROOT={q(results_root)} " + " ".join(q(arg) for arg in args)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="place_dual_shoes")
    parser.add_argument("--task-config", default="demo_clean")
    parser.add_argument("--expert-data-num", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-gpu", default="1")
    parser.add_argument("--eval-gpu", default="1")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-epochs", type=int, default=200)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--checkpoint", type=int, default=200)
    parser.add_argument("--test-num", type=int, default=100)
    parser.add_argument("--results-root", default="results_dual_shoes_50demo_trajectory_eval100")
    parser.add_argument("--output", default="scripts/generated_dual_shoes_trajectory_commands.sh")
    parser.add_argument("--include-delta-changed", action="store_true")
    args = parser.parse_args()

    root = Path.cwd()
    results_root = args.results_root
    if not os.path.isabs(results_root):
        results_root = str(root / results_root)
    zarr_path = root / "data" / f"{args.task}-{args.task_config}-{args.expert_data_num}-pi3-20-5.zarr"

    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Generated commands for public-50 Place Dual Shoes trajectory-supervision variants.",
        "# The script preprocesses only if the zarr is missing, trains only if a checkpoint is missing,",
        "# and evaluates only after the requested checkpoint exists.",
        "",
        f"ZARR_PATH={q(str(zarr_path))}",
        'if [ ! -d "${ZARR_PATH}" ]; then',
        f"  bash process_data.sh {q(args.task)} {q(args.task_config)} {args.expert_data_num} {q(args.train_gpu)}",
        "else",
        '  printf "Zarr already exists: %s\\n" "${ZARR_PATH}"',
        "fi",
        "",
    ]

    for variant in variant_specs(args.include_delta_changed):
        ckpt = root / "checkpoints" / f"{args.task}_{variant.checkpoint_tag}_{args.expert_data_num}" / f"{args.checkpoint}.ckpt"
        lines.extend(
            [
                f"# variant={variant.name}",
                f"CKPT_PATH={q(str(ckpt))}",
                'if [ ! -f "${CKPT_PATH}" ]; then',
                f"  {train_command(variant, args.task, args.task_config, args.expert_data_num, args.seed, args.train_gpu, args.batch_size, args.num_epochs, args.checkpoint_every)}",
                "else",
                '  printf "Checkpoint already exists: %s\\n" "${CKPT_PATH}"',
                "fi",
                'if [ -f "${CKPT_PATH}" ]; then',
                f"  {eval_command(variant, args.task, args.task_config, args.expert_data_num, args.seed, args.eval_gpu, args.checkpoint, args.test_num, results_root)}",
                "else",
                '  printf "Skipping eval because checkpoint is still missing: %s\\n" "${CKPT_PATH}"',
                "fi",
                "",
            ]
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    output.chmod(0o755)
    print(f"Wrote trajectory commands to {output}")


if __name__ == "__main__":
    main()
