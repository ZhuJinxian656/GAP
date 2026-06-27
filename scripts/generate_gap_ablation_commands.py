#!/usr/bin/env python
"""Generate GAP interface-ablation training commands without executing them."""
from __future__ import annotations

import argparse
import os
import shlex
from itertools import product
from typing import Dict, List


VARIANT_OVERRIDES: Dict[str, List[str]] = {
    "vanilla": [
        "latent_mode=pi3_full",
        "use_future_loss=true",
        "future_target_mode=pi3_full",
        "use_pi3_features=true",
        "policy.use_pi3_features=true",
    ],
    "dino_only": [
        "latent_mode=dino_only",
        "use_future_loss=false",
        "future_target_mode=none",
        "use_pi3_features=false",
        "policy.use_pi3_features=false",
    ],
    "no_future": [
        "latent_mode=pi3_full",
        "use_future_loss=false",
        "future_target_mode=none",
        "use_pi3_features=true",
        "policy.use_pi3_features=true",
    ],
    "pi3_pooled": [
        "latent_mode=pi3_pooled",
        "use_future_loss=true",
        "future_target_mode=pi3_pooled",
        "use_pi3_features=true",
        "policy.use_pi3_features=true",
    ],
    "pi3_compressed": [
        "latent_mode=pi3_compressed",
        "use_future_loss=true",
        "future_target_mode=pi3_compressed",
        "use_pi3_features=true",
        "policy.use_pi3_features=true",
    ],
    "pi3_random": [
        "latent_mode=pi3_random_tokens",
        "use_future_loss=true",
        "future_target_mode=pi3_random_tokens",
        "use_pi3_features=true",
        "policy.use_pi3_features=true",
    ],
    "pi3_dropout": [
        "latent_mode=pi3_token_dropout",
        "use_future_loss=true",
        "future_target_mode=pi3_full",
        "use_pi3_features=true",
        "policy.use_pi3_features=true",
    ],
    "object_hand": [
        "latent_mode=pi3_object_hand",
        "use_future_loss=true",
        "future_target_mode=pi3_object_hand",
        "use_pi3_features=true",
        "policy.use_pi3_features=true",
    ],
    "background": [
        "latent_mode=pi3_background",
        "use_future_loss=true",
        "future_target_mode=pi3_background",
        "use_pi3_features=true",
        "policy.use_pi3_features=true",
    ],
}


def quote_list(items: List[str]) -> str:
    return " ".join(shlex.quote(item) for item in items)


def command_for(
    task: str,
    setting: str,
    expert_data_num: int,
    seed: int,
    gpu: str,
    batch_size: int,
    num_epochs: int,
    checkpoint_every: int,
    variant: str,
) -> str:
    overrides = VARIANT_OVERRIDES[variant]
    exp_name = f"{task}_{setting}_{expert_data_num}_{variant}_seed{seed}"
    overrides = overrides + [
        f"exp_name={exp_name}",
        f"logging.name={exp_name}",
    ]
    return (
        f"bash train.sh {shlex.quote(task)} {shlex.quote(setting)} {expert_data_num} "
        f"{seed} {shlex.quote(gpu)} {batch_size} {num_epochs} {checkpoint_every} "
        f"{quote_list(overrides)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate GAP ablation commands.")
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--settings", nargs="+", default=["demo_clean"])
    parser.add_argument("--expert-data-num", type=int, default=100)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--gpus", nargs="+", default=["0"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-epochs", type=int, default=300)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--variants", nargs="+", default=["vanilla", "dino_only", "no_future"])
    parser.add_argument("--output", default="scripts/generated_ablation_commands.sh")
    args = parser.parse_args()

    unknown = [variant for variant in args.variants if variant not in VARIANT_OVERRIDES]
    if unknown:
        raise SystemExit(f"Unknown variants: {unknown}. Expected one of {sorted(VARIANT_OVERRIDES)}")

    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Generated commands only; review data availability before running object_hand/background variants.",
        "# Each command uses a single GPU selected round-robin from --gpus.",
        "",
    ]

    run_id = 0
    for task, setting, seed, variant in product(args.tasks, args.settings, args.seeds, args.variants):
        gpu = str(args.gpus[run_id % len(args.gpus)])
        cmd = command_for(
            task=task,
            setting=setting,
            expert_data_num=args.expert_data_num,
            seed=seed,
            gpu=gpu,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            checkpoint_every=args.checkpoint_every,
            variant=variant,
        )
        lines.append(f"# run {run_id}: task={task} setting={setting} seed={seed} variant={variant} gpu={gpu}")
        lines.append(cmd)
        lines.append("")
        run_id += 1

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    os.chmod(args.output, 0o755)
    print(f"Wrote {run_id} commands to {args.output}")


if __name__ == "__main__":
    main()
