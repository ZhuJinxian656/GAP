"""
Training script for GAP.
"""
if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import sys
import hydra
import torch
from omegaconf import OmegaConf
import pathlib

GAP_ROOT = str(pathlib.Path(__file__).parent.parent)

sys.path.append(GAP_ROOT)
sys.path.append(os.path.join(GAP_ROOT, "gap_policy"))

from torch.utils.data import DataLoader
import copy

import wandb
from tqdm import tqdm
import numpy as np
from termcolor import cprint
import random

from gap_policy.dataset.gap_dataset import GAPDataset
from gap_policy.common.pytorch_util import dict_apply
from gap_policy.model.diffusion.ema_model import EMAModel
from gap_policy.model.common.lr_scheduler import get_scheduler

OmegaConf.register_new_resolver("eval", eval, replace=True)


@hydra.main(
    version_base=None,
    config_path="../gap_policy/config",
    config_name="GAP",
)
def main(cfg: OmegaConf):
    # Set random seed
    seed = cfg.training.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Get task name and setting
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

    cprint(f"[GAP Training]", "cyan", attrs=["bold"])
    cprint(f"  Task: {task_name}", "cyan")
    cprint(f"  Setting: {setting}", "cyan")
    cprint(f"  Expert data: {expert_data_num}", "cyan")
    cprint(f"  Zarr path: {zarr_path}", "cyan")
    cprint(f"  Seed: {seed}", "cyan")

    policy_cfg = cfg.policy
    coupling_cfg = policy_cfg.get("coupling", {})
    coupling_enabled = bool(coupling_cfg.get("enabled", False)) if coupling_cfg is not None else False
    use_triadic_token = bool(policy_cfg.get("use_triadic_token", False)) or coupling_enabled
    triadic_mode = policy_cfg.get("triadic_mode", "disabled")
    if use_triadic_token and triadic_mode == "disabled":
        triadic_mode = coupling_cfg.get("feature_mode", "proprio_only_fallback")

    # Create dataset
    dataset = GAPDataset(
        zarr_path=zarr_path,
        horizon=cfg.horizon,
        pad_before=cfg.n_obs_steps - 1,
        pad_after=cfg.n_action_steps - 1,
        seed=seed,
        val_ratio=cfg.training.get("val_ratio", 0.0),
        max_train_episodes=expert_data_num,
        task_name=task_name,
        use_pi3_features=cfg.get("use_pi3_features", True),
        model_3d=model_3d,
        use_triadic_token=use_triadic_token,
        triadic_mode=triadic_mode,
        latent_mode=cfg.get("latent_mode", policy_cfg.get("latent_mode", "pi3_full")),
        use_future_loss=cfg.get("use_future_loss", policy_cfg.get("use_future_loss", True)),
        future_target_mode=cfg.get("future_target_mode", policy_cfg.get("future_target_mode", "pi3_full")),
        use_interaction_field=policy_cfg.get("use_interaction_field", False),
        interaction_field_mode=policy_cfg.get("interaction_field", {}).get("mode", "disabled"),
    )

    # Get normalizer
    normalizer = dataset.get_normalizer()

    # Create dataloader
    train_dataloader = DataLoader(
        dataset,
        batch_size=cfg.dataloader.batch_size,
        num_workers=cfg.dataloader.num_workers,
        shuffle=cfg.dataloader.shuffle,
        pin_memory=cfg.dataloader.pin_memory,
        persistent_workers=cfg.dataloader.persistent_workers,
    )
    val_dataloader = None
    if cfg.training.get("val_ratio", 0.0) > 0:
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=cfg.val_dataloader.batch_size,
            num_workers=cfg.val_dataloader.num_workers,
            shuffle=cfg.val_dataloader.shuffle,
            pin_memory=cfg.val_dataloader.pin_memory,
            persistent_workers=cfg.val_dataloader.persistent_workers,
        )

    # Create policy
    cprint("\nCreating GAP policy...", "green")
    policy = hydra.utils.instantiate(cfg.policy)

    # Set normalizer
    policy.set_normalizer(normalizer)

    # Move to device
    device = torch.device(cfg.training.device)
    policy = policy.to(device)

    # Create EMA model
    ema: EMAModel = None
    if cfg.training.use_ema:
        ema_policy = copy.deepcopy(policy)
        ema = hydra.utils.instantiate(cfg.ema, model=ema_policy)

    # Create optimizer
    optimizer = hydra.utils.instantiate(cfg.optimizer, params=policy.parameters())

    # Create learning rate scheduler
    lr_scheduler = get_scheduler(
        cfg.training.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.training.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * cfg.training.num_epochs) // cfg.training.gradient_accumulate_every,
    )

    # Initialize wandb
    wandb_mode = cfg.logging.get("mode", "online")
    use_wandb = (not cfg.training.debug) and wandb_mode != "disabled"
    if use_wandb:
        wandb.init(
            project=cfg.logging.project,
            name=f"{cfg.name}_{task_name}_{setting}_{expert_data_num}",
            config=OmegaConf.to_container(cfg, resolve=True),
            mode=wandb_mode,
        )
    if cfg.training.debug:
        cfg.training.num_epochs = 100
        cfg.training.max_train_steps = 10
        cfg.training.max_val_steps = 3
        cfg.training.checkpoint_every = cfg.training.num_epochs
        cfg.training.val_every = 1

    # Training loop
    cprint("\nStarting training...", "green", attrs=["bold"])
    global_step = 0

    for epoch in range(cfg.training.num_epochs):
        policy.train()

        epoch_loss = 0.0
        epoch_loss_count = 0
        with tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{cfg.training.num_epochs}") as pbar:
            for batch_idx, batch in enumerate(pbar):
                if cfg.training.max_train_steps is not None and batch_idx >= cfg.training.max_train_steps:
                    break
                # Move batch to device
                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))

                # Forward pass
                loss, loss_dict = policy.compute_loss(batch)

                # Backward pass
                loss.backward()

                # Gradient accumulation
                if (batch_idx + 1) % cfg.training.gradient_accumulate_every == 0:
                    # Clip gradients
                    if cfg.training.get("clip_grad_norm", None):
                        torch.nn.utils.clip_grad_norm_(
                            policy.parameters(),
                            cfg.training.clip_grad_norm
                        )

                    # Update parameters
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                    # Update EMA
                    if ema is not None:
                        ema.step(policy)

                    global_step += 1

                # Logging
                epoch_loss += loss.item()
                epoch_loss_count += 1
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

                # Log to wandb
                if use_wandb and (batch_idx % 10 == 0):
                    log_dict = {
                        "train/loss": loss.item(),
                        "train/lr": lr_scheduler.get_last_lr()[0],
                        "train/epoch": epoch,
                        "train/global_step": global_step,
                    }
                    # Add loss_dict items with train/ prefix
                    for key, value in loss_dict.items():
                        log_dict[f"train/{key}"] = value
                    wandb.log(log_dict, step=global_step)

        # Epoch summary
        avg_epoch_loss = epoch_loss / max(1, epoch_loss_count)
        cprint(f"Epoch {epoch+1} - Avg Loss: {avg_epoch_loss:.4f}", "yellow")

        if use_wandb:
            wandb.log({
                "train/epoch_loss": avg_epoch_loss,
                "train/epoch": epoch,
            }, step=global_step)

        if (
            val_dataloader is not None
            and cfg.training.val_every is not None
            and (epoch + 1) % cfg.training.val_every == 0
        ):
            policy.eval()
            val_loss = 0.0
            val_loss_count = 0
            val_metrics = {}
            with torch.no_grad():
                with tqdm(val_dataloader, desc=f"Val {epoch+1}/{cfg.training.num_epochs}") as pbar:
                    for val_idx, batch in enumerate(pbar):
                        if cfg.training.max_val_steps is not None and val_idx >= cfg.training.max_val_steps:
                            break
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        loss, loss_dict = policy.compute_loss(batch)
                        val_loss += loss.item()
                        val_loss_count += 1
                        for key, value in loss_dict.items():
                            if isinstance(value, (int, float)):
                                val_metrics.setdefault(key, []).append(float(value))
                        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            avg_val_loss = val_loss / max(1, val_loss_count)
            avg_val_metrics = {
                key: float(np.mean(values))
                for key, values in val_metrics.items()
                if values
            }
            cprint(f"Epoch {epoch+1} - Val Loss: {avg_val_loss:.4f}", "yellow")
            if avg_val_metrics:
                cprint(
                    "  Val metrics: "
                    + ", ".join(f"{key}={value:.6f}" for key, value in sorted(avg_val_metrics.items())),
                    "yellow",
                )
            if use_wandb:
                log_dict = {
                    "val/loss": avg_val_loss,
                    "val/epoch": epoch,
                }
                for key, value in avg_val_metrics.items():
                    log_dict[f"val/{key}"] = value
                wandb.log(log_dict, step=global_step)

        # Save checkpoint
        save_ckpt = bool(cfg.get("checkpoint", {}).get("save_ckpt", True))
        checkpoint_tag = cfg.get("checkpoint_tag", None)
        checkpoint_dir_name = (
            f"{task_name}_{checkpoint_tag}_{expert_data_num}"
            if checkpoint_tag
            else f"{task_name}_{setting}_{expert_data_num}"
        )
        checkpoint_dir = os.path.join(os.getcwd(), "checkpoints", checkpoint_dir_name)
        if save_ckpt and (epoch + 1) % cfg.training.checkpoint_every == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"{epoch+1}.ckpt")
            os.makedirs(checkpoint_dir, exist_ok=True)

            checkpoint = {
                "epoch": epoch,
                "global_step": global_step,
                "model": policy.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "cfg": OmegaConf.to_container(cfg, resolve=True),
                'normalizer': normalizer.state_dict(),
            }

            if ema is not None:
                checkpoint["ema"] = ema.averaged_model.state_dict()

            torch.save(checkpoint, checkpoint_path)
            cprint(f"Saved checkpoint to {checkpoint_path}", "green")

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
