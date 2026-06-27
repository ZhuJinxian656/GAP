"""
Distributed training script for GAP.

This keeps scripts/train.py as the single-GPU entry point and adds a DDP path
for long baseline runs.
"""
if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import copy
import os
import pathlib
import random

import hydra
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from omegaconf import OmegaConf
from termcolor import cprint
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import wandb

GAP_ROOT = str(pathlib.Path(__file__).parent.parent)

import sys
sys.path.append(GAP_ROOT)
sys.path.append(os.path.join(GAP_ROOT, "gap_policy"))

from gap_policy.common.pytorch_util import dict_apply
from gap_policy.dataset.gap_dataset import GAPDataset
from gap_policy.model.diffusion.ema_model import EMAModel
from gap_policy.model.common.lr_scheduler import get_scheduler

OmegaConf.register_new_resolver("eval", eval, replace=True)


class GAPLossWrapper(nn.Module):
    def __init__(self, policy: nn.Module):
        super().__init__()
        self.policy = policy

    def forward(self, batch):
        return self.policy.compute_loss(batch)


def setup_distributed():
    if "RANK" not in os.environ:
        return 0, 0, 1

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return rank, local_rank, world_size


def cleanup_distributed(world_size: int):
    if world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()


def reduce_mean(value: torch.Tensor, world_size: int) -> torch.Tensor:
    if world_size == 1:
        return value.detach()
    reduced = value.detach().clone()
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    reduced /= world_size
    return reduced


def rank0_print(rank: int, message: str, *args, **kwargs):
    if rank == 0:
        cprint(message, *args, **kwargs)


@hydra.main(
    version_base=None,
    config_path="../gap_policy/config",
    config_name="GAP",
)
def main(cfg: OmegaConf):
    rank, local_rank, world_size = setup_distributed()
    is_rank0 = rank == 0

    try:
        seed = cfg.training.seed
        torch.manual_seed(seed + rank)
        np.random.seed(seed + rank)
        random.seed(seed + rank)

        task_name = cfg.task_name
        setting = cfg.get("setting", "demo_clean")
        expert_data_num = cfg.expert_data_num
        observation_chunk = cfg.observation_chunk
        interval = cfg.interval
        model_3d = cfg.model_3d

        data_root = cfg.get("data_root", os.environ.get("GAP_DATA_ROOT", "data"))
        if not os.path.isabs(data_root):
            data_root = os.path.join(GAP_ROOT, data_root)
        zarr_path = os.path.join(
            data_root,
            f"{task_name}-{setting}-{expert_data_num}-{model_3d}-{observation_chunk}-{interval}.zarr",
        )

        if is_rank0:
            cprint("[GAP DDP Training]", "cyan", attrs=["bold"])
            cprint(f"  Task: {task_name}", "cyan")
            cprint(f"  Setting: {setting}", "cyan")
            cprint(f"  Expert data: {expert_data_num}", "cyan")
            cprint(f"  Zarr path: {zarr_path}", "cyan")
            cprint(f"  Seed: {seed}", "cyan")
            cprint(f"  World size: {world_size}", "cyan")
            cprint(f"  Per-GPU batch size: {cfg.dataloader.batch_size}", "cyan")
            cprint(f"  Global batch size: {cfg.dataloader.batch_size * world_size}", "cyan")

        dataset = GAPDataset(
            zarr_path=zarr_path,
            horizon=cfg.horizon,
            pad_before=cfg.n_obs_steps - 1,
            pad_after=cfg.n_action_steps - 1,
            seed=seed,
            val_ratio=0.0,
            max_train_episodes=expert_data_num,
            task_name=task_name,
            use_pi3_features=cfg.get("use_pi3_features", True),
            model_3d=model_3d,
            use_triadic_token=cfg.policy.get("use_triadic_token", False),
            triadic_mode=cfg.policy.get("triadic_mode", "disabled"),
        )
        normalizer = dataset.get_normalizer()

        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=cfg.dataloader.shuffle,
            seed=seed,
            drop_last=False,
        ) if world_size > 1 else None

        train_dataloader = DataLoader(
            dataset,
            batch_size=cfg.dataloader.batch_size,
            num_workers=cfg.dataloader.num_workers,
            shuffle=cfg.dataloader.shuffle if sampler is None else False,
            sampler=sampler,
            pin_memory=cfg.dataloader.pin_memory,
            persistent_workers=cfg.dataloader.persistent_workers and cfg.dataloader.num_workers > 0,
        )

        if is_rank0:
            cprint("\nCreating GAP policy...", "green")
        policy = hydra.utils.instantiate(cfg.policy)
        policy.set_normalizer(normalizer)

        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        policy = policy.to(device)
        ddp_model = DDP(
            GAPLossWrapper(policy),
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
            find_unused_parameters=cfg.training.get("ddp_find_unused_parameters", True),
        ) if world_size > 1 else GAPLossWrapper(policy)

        ema: EMAModel = None
        if cfg.training.use_ema and is_rank0:
            ema_policy = copy.deepcopy(policy)
            ema = hydra.utils.instantiate(cfg.ema, model=ema_policy)

        optimizer = hydra.utils.instantiate(cfg.optimizer, params=ddp_model.parameters())
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(len(train_dataloader) * cfg.training.num_epochs) // cfg.training.gradient_accumulate_every,
        )

        wandb_mode = cfg.logging.get("mode", "online")
        wandb_enabled = (not cfg.training.debug) and wandb_mode != "disabled"
        if is_rank0 and wandb_enabled:
            wandb.init(
                project=cfg.logging.project,
                name=f"{cfg.name}_{task_name}_{setting}_{expert_data_num}_ddp{world_size}",
                config=OmegaConf.to_container(cfg, resolve=True),
                mode=wandb_mode,
            )

        if cfg.training.debug:
            cfg.training.num_epochs = 100
            cfg.training.max_train_steps = 10
            cfg.training.max_val_steps = 3
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1

        if is_rank0:
            cprint("\nStarting DDP training...", "green", attrs=["bold"])
        global_step = 0

        for epoch in range(cfg.training.num_epochs):
            if sampler is not None:
                sampler.set_epoch(epoch)
            ddp_model.train()

            epoch_loss = 0.0
            step_count = 0
            iterator = train_dataloader
            if is_rank0:
                iterator = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{cfg.training.num_epochs}")

            for batch_idx, batch in enumerate(iterator):
                if cfg.training.max_train_steps is not None and batch_idx >= cfg.training.max_train_steps:
                    break

                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                loss, loss_dict = ddp_model(batch)
                loss.backward()

                if (batch_idx + 1) % cfg.training.gradient_accumulate_every == 0:
                    if cfg.training.get("clip_grad_norm", None):
                        torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), cfg.training.clip_grad_norm)

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                    if ema is not None:
                        ema.step(policy)

                    global_step += 1

                reduced_loss = reduce_mean(loss, world_size)
                if is_rank0:
                    loss_value = reduced_loss.item()
                    epoch_loss += loss_value
                    step_count += 1
                    iterator.set_postfix({"loss": f"{loss_value:.4f}"})

                should_log_metrics = wandb_enabled and (batch_idx % 10 == 0)
                reduced_loss_dict = None
                if should_log_metrics:
                    reduced_loss_dict = {}
                    for key, value in loss_dict.items():
                        if isinstance(value, torch.Tensor):
                            metric_tensor = value.float()
                        else:
                            metric_tensor = torch.tensor(value, device=device, dtype=torch.float32)
                        reduced_loss_dict[key] = reduce_mean(metric_tensor, world_size).item()

                if is_rank0:
                    if should_log_metrics:
                        log_dict = {
                            "train/loss": loss_value,
                            "train/lr": lr_scheduler.get_last_lr()[0],
                            "train/epoch": epoch,
                            "train/global_step": global_step,
                        }
                        for key, metric_value in reduced_loss_dict.items():
                            log_dict[f"train/{key}"] = metric_value
                        wandb.log(log_dict, step=global_step)

            if is_rank0:
                avg_epoch_loss = epoch_loss / max(step_count, 1)
                cprint(f"Epoch {epoch+1} - Avg Loss: {avg_epoch_loss:.4f}", "yellow")

                if wandb_enabled:
                    wandb.log({
                        "train/epoch_loss": avg_epoch_loss,
                        "train/epoch": epoch,
                    }, step=global_step)

                checkpoint_dir = os.path.join(os.getcwd(), "checkpoints", f"{task_name}_{setting}_{expert_data_num}")
                if (epoch + 1) % cfg.training.checkpoint_every == 0:
                    checkpoint_path = os.path.join(checkpoint_dir, f"{epoch+1}.ckpt")
                    os.makedirs(checkpoint_dir, exist_ok=True)

                    checkpoint = {
                        "epoch": epoch,
                        "global_step": global_step,
                        "model": policy.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "cfg": OmegaConf.to_container(cfg, resolve=True),
                        "normalizer": normalizer.state_dict(),
                    }
                    if ema is not None:
                        checkpoint["ema"] = ema.averaged_model.state_dict()

                    torch.save(checkpoint, checkpoint_path)
                    cprint(f"Saved checkpoint to {checkpoint_path}", "green")

            if world_size > 1:
                dist.barrier()

        if is_rank0 and wandb_enabled:
            wandb.finish()
    finally:
        cleanup_distributed(world_size)


if __name__ == "__main__":
    main()
