import sys
import os
import torch
import numpy as np
from termcolor import cprint

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, "thirdparty"))
from pi3.models.pi3 import Pi3

from gap_policy.policy.gap import GAPPolicy
from gap_policy.model.vision.dinov3_encoder import DINOV3
from gap_policy.latent_modes import mask_key_for_mode
from gap_policy.triadic import TriadicConfig, build_triadic_state


PRETRAINED_ROOT = os.environ.get("GAP_PRETRAINED_ROOT", "pretrained")
DEFAULT_PI3_PATH = os.path.join(PRETRAINED_ROOT, "Pi3")
DEFAULT_DINOV3_WEIGHTS = os.path.join(
    PRETRAINED_ROOT,
    "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
)


def resolve_repo_path(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(current_dir, path)


class GAPPolicyWrapper:

    def __init__(
        self,
        ckpt_path: str,
        device: str = "cuda",
        debug: bool = False,
    ):
        self.device = device
        self.debug = debug

        cprint(f"[GAP] Initializing deployment policy", "cyan")
        cprint(f"[GAP] Device: {device}", "cyan")

        # Load checkpoint
        cprint(f"[GAP] Loading checkpoint from {ckpt_path}", "cyan")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

        class SubscriptableNamespace(dict):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                for k, v in self.items():
                    if isinstance(v, dict):
                        self[k] = SubscriptableNamespace(v)
                    elif isinstance(v, list):
                        self[k] = [SubscriptableNamespace(x) if isinstance(x, dict) else x for x in v]
            def __getattr__(self, key):
                try:
                    return self[key]
                except KeyError:
                    raise AttributeError(key)
            def __setattr__(self, key, value):
                self[key] = value
            def __delattr__(self, key):
                del self[key]

        self.cfg = ckpt.get("cfg")
        if isinstance(self.cfg, dict):
            self.cfg = SubscriptableNamespace(self.cfg)

        if self.cfg is None:
            raise ValueError("Checkpoint does not contain 'cfg' key")

        # Get policy configuration
        policy_cfg = self.cfg.policy

        cprint(f"[GAP] Creating GAP policy model", "cyan")
        cprint(f"[GAP] DinoV3 model: {policy_cfg.dinov3_model_name}", "cyan")
        cprint(f"[GAP] Horizon: {policy_cfg.horizon}", "cyan")
        cprint(f"[GAP] N obs steps: {policy_cfg.n_obs_steps}", "cyan")
        cprint(f"[GAP] N action steps: {policy_cfg.n_action_steps}", "cyan")

        # Create noise scheduler from config
        noise_scheduler_cfg = policy_cfg.noise_scheduler
        target = noise_scheduler_cfg["_target_"]
        module_path, class_name = target.rsplit(".", 1)
        import importlib
        module = importlib.import_module(module_path)
        scheduler_class = getattr(module, class_name)

        # Create scheduler with config parameters
        scheduler_params = {k: v for k, v in noise_scheduler_cfg.items() if k != "_target_"}
        noise_scheduler = scheduler_class(**scheduler_params)
        policy_cfg.noise_scheduler = noise_scheduler
        cprint(f"[GAP] Created {class_name} noise scheduler", "cyan")

        # Create GAP model
        self.policy_model = GAPPolicy(
            **policy_cfg
        )

        # Load model weights (use EMA if available)
        if "ema" in ckpt and ckpt["ema"] is not None:
            cprint(f"[GAP] Loading EMA model weights", "cyan")
            self.policy_model.load_state_dict(ckpt["ema"])
        elif "model" in ckpt:
            cprint(f"[GAP] Loading model weights", "cyan")
            self.policy_model.load_state_dict(ckpt["model"])
        else:
            raise ValueError("Checkpoint does not contain model weights")

        self.policy_model.to(device)
        self.policy_model.eval()

        # Load normalizer
        if "normalizer" in ckpt:
            class FakedNormalizer:
                def __init__(self, state_dict):
                    self._state_dict = state_dict
                def state_dict(self):
                    return self._state_dict
            normalizer = FakedNormalizer(ckpt["normalizer"])
            self.policy_model.set_normalizer(normalizer)
            cprint(f"[GAP] Loaded normalizer from checkpoint", "cyan")
        else:
            raise ValueError(
                "Checkpoint does not contain 'normalizer'. "
                "Use checkpoints saved by scripts/train.py or add the training normalizer state to the checkpoint."
            )

        cprint(f"[GAP] Policy loaded successfully!", "green")

        # Load DinoV3 feature extractor
        cprint(f"[GAP] Loading DinoV3 feature extractor...", "cyan")
        dinov3_repo_dir = policy_cfg.get("dinov3_repo_dir", "thirdparty/dinov3")
        dinov3_repo_path = resolve_repo_path(dinov3_repo_dir)
        if not os.path.exists(dinov3_repo_path):
            fallback_repo_path = os.path.join(current_dir, "thirdparty/dinov3")
            if os.path.exists(fallback_repo_path):
                dinov3_repo_path = fallback_repo_path
        dinov3_weights_path = os.environ.get(
            "DINOV3_WEIGHTS_PATH",
            policy_cfg.get("dinov3_weights_path", DEFAULT_DINOV3_WEIGHTS),
        )
        dinov3_weights_file = resolve_repo_path(dinov3_weights_path)
        if not os.path.exists(dinov3_weights_file):
            raise FileNotFoundError(f"DINOv3 weights not found: {dinov3_weights_file}")
        self.dinov3_model = DINOV3(
            model_name=policy_cfg.dinov3_model_name,
            repo_dir=dinov3_repo_path,
            weights_path=dinov3_weights_file,
            freeze=True
        ).to(device).eval()
        cprint(f"[GAP] DinoV3 feature extractor loaded", "green")

        # Load Pi3 feature extractor (if enabled)
        self.use_pi3 = policy_cfg.get('use_pi3_features', False)
        if self.use_pi3:
            cprint(f"[GAP] Loading Pi3 feature extractor...", "cyan")
            pi3_model_name_or_path = os.environ.get(
                "PI3_MODEL_NAME_OR_PATH",
                os.environ.get(
                    "PI3_MODEL_PATH",
                    policy_cfg.get("pi3_model_name_or_path", DEFAULT_PI3_PATH),
                ),
            )
            pi3_model_name_or_path = resolve_repo_path(pi3_model_name_or_path)
            self.pi3_model = Pi3.from_pretrained(pi3_model_name_or_path).to(device).eval()
            cprint(f"[GAP] Pi3 feature extractor loaded", "green")
        else:
            self.pi3_model = None

        # Store config
        self.n_action_steps = policy_cfg.n_action_steps
        self.state_dim = policy_cfg.state_dim
        self.use_triadic_token = policy_cfg.get("use_triadic_token", False)
        self.triadic_config = TriadicConfig(
            mode=policy_cfg.get("triadic_mode", "disabled"),
            include_grippers=policy_cfg.get("triadic_include_grippers", True),
        )
        self.latent_mode = getattr(self.policy_model, "latent_mode", policy_cfg.get("latent_mode", "pi3_full"))
        self.latent_mask_key = mask_key_for_mode(self.latent_mode)
        self.eef_mask_radius_tokens = float(os.environ.get("GAP_EEF_MASK_RADIUS_TOKENS", "2.5"))
        self.use_interaction_field = bool(policy_cfg.get("use_interaction_field", False))
        interaction_cfg = policy_cfg.get("interaction_field", {})
        self.interaction_field_mode = interaction_cfg.get("mode", "disabled")
        self.dino_eef_mask_radius_tokens = float(os.environ.get("GAP_DINO_EEF_MASK_RADIUS_TOKENS", "2.5"))
        self.dino_pair_mask_radius_tokens = float(os.environ.get("GAP_DINO_PAIR_MASK_RADIUS_TOKENS", "2.5"))

    def reset(self):
        """Reset policy state between episodes (GAP is stateless)"""
        if self.debug:
            cprint("[GAP] Policy reset (stateless)", "cyan")

    def extract_dinov3_features(self, rgb_image: np.ndarray) -> torch.Tensor:
        """
        Extract DinoV3 features from RGB image

        Args:
            rgb_image: numpy array [H, W, 3] in RGB format, range [0, 255]

        Returns:
            features: [1, N_patches, D] tensor
        """
        # Convert to tensor and normalize for ImageNet
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)

        # RGB to tensor [1, 3, H, W] and normalize to [0, 1]
        img_tensor = torch.from_numpy(rgb_image).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)

        # Apply ImageNet normalization
        img_tensor = (img_tensor - mean) / std

        # Extract features
        with torch.no_grad():
            features = self.dinov3_model(img_tensor)  # [1, N_patches, D]

        return features

    def extract_pi3_features(self, rgb_image: np.ndarray) -> torch.Tensor:
        """
        Extract Pi3 features from RGB image

        Args:
            rgb_image: numpy array [H, W, 3] in RGB format, range [0, 255]

        Returns:
            features: [1, N_patches, 1024] tensor
        """
        # Convert RGB to tensor
        img_tensor = torch.from_numpy(rgb_image).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(0).to(self.device)  # [1, 1, 3, H, W]

        # Resize to be divisible by 14 (Pi3 patch size)
        H, W = rgb_image.shape[:2]
        target_h = round(H / 14) * 14
        target_w = round(W / 14) * 14
        target_h = max(target_h, 14)
        target_w = max(target_w, 14)

        if H != target_h or W != target_w:
            img_tensor = torch.nn.functional.interpolate(
                img_tensor.squeeze(0), size=(target_h, target_w), mode='bilinear', align_corners=False
            ).unsqueeze(0)

        # Normalize
        img_tensor = (img_tensor - self.pi3_model.image_mean) / self.pi3_model.image_std

        # Extract features
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=dtype):
                N, C, H_new, W_new = img_tensor.shape[1:]
                imgs_flat = img_tensor.squeeze(0)  # [N, C, H, W]

                # Encode
                hidden = self.pi3_model.encoder(imgs_flat, is_training=True)
                if isinstance(hidden, dict):
                    hidden = hidden["x_norm_patchtokens"]

                # Decode
                hidden, pos = self.pi3_model.decode(hidden, N, H_new, W_new)

                # Extract point features
                point_hidden = self.pi3_model.point_decoder(hidden, xpos=pos)  # [N, hw, 1024]
                point_hidden = point_hidden[:, self.pi3_model.patch_start_idx:].float()  # [N, num_patches, 1024]

        return point_hidden.unsqueeze(0)  # [1, N, num_patches, 1024]

    def _project_point_to_head_camera(self, point_world: np.ndarray, raw_observation: dict) -> tuple[np.ndarray, bool]:
        camera_obs = raw_observation["observation"]["head_camera"]
        intrinsic = np.asarray(camera_obs["intrinsic_cv"], dtype=np.float32)
        extrinsic = np.asarray(camera_obs["extrinsic_cv"], dtype=np.float32)
        point_world = np.asarray(point_world, dtype=np.float32).reshape(3)
        point_cam = extrinsic[:, :3] @ point_world + extrinsic[:, 3]
        if not np.isfinite(point_cam).all() or point_cam[2] <= 1e-6:
            return np.zeros(2, dtype=np.float32), False
        uvw = intrinsic @ point_cam
        uv = uvw[:2] / uvw[2]
        return uv.astype(np.float32), bool(np.isfinite(uv).all())

    def _eef_region_mask(self, rgb_image: np.ndarray, raw_observation: dict, *, complement: bool) -> torch.Tensor:
        if raw_observation is None:
            raise KeyError(
                f"latent_mode={self.latent_mode!r} requires raw RoboTwin observation with "
                "head-camera calibration and left/right endpose fields."
            )
        try:
            left = np.asarray(raw_observation["endpose"]["left_endpose"], dtype=np.float32)[:3]
            right = np.asarray(raw_observation["endpose"]["right_endpose"], dtype=np.float32)[:3]
        except KeyError as exc:
            raise KeyError(
                f"latent_mode={self.latent_mode!r} requires raw_observation['endpose'] with "
                "'left_endpose' and 'right_endpose'."
            ) from exc

        image_h, image_w = rgb_image.shape[:2]
        grid_h = int(self.policy_model.pi3_grid_height)
        grid_w = int(self.policy_model.pi3_grid_width)
        yy, xx = np.meshgrid(np.arange(grid_h, dtype=np.float32), np.arange(grid_w, dtype=np.float32), indexing="ij")
        mask = np.zeros((grid_h, grid_w), dtype=bool)

        for point in (left, right):
            uv, valid = self._project_point_to_head_camera(point, raw_observation)
            if not valid:
                continue
            if uv[0] < 0 or uv[0] >= image_w or uv[1] < 0 or uv[1] >= image_h:
                continue
            token_x = (uv[0] / image_w) * grid_w - 0.5
            token_y = (uv[1] / image_h) * grid_h - 0.5
            dist2 = (xx - token_x) ** 2 + (yy - token_y) ** 2
            mask |= dist2 <= self.eef_mask_radius_tokens ** 2

        if complement:
            mask = ~mask
        mask = mask.astype(np.float32).reshape(1, 1, grid_h * grid_w)
        return torch.from_numpy(mask).to(device=self.device)

    def _disk_token_mask(
        self,
        token_uv: np.ndarray,
        valid: bool,
        grid_h: int,
        grid_w: int,
        radius_tokens: float,
    ) -> np.ndarray:
        mask = np.zeros((grid_h, grid_w), dtype=bool)
        if not valid:
            return mask
        yy, xx = np.meshgrid(
            np.arange(grid_h, dtype=np.float32),
            np.arange(grid_w, dtype=np.float32),
            indexing="ij",
        )
        dist2 = (xx - token_uv[0]) ** 2 + (yy - token_uv[1]) ** 2
        mask = dist2 <= radius_tokens ** 2
        return mask

    def _segment_token_mask(
        self,
        left_token_uv: np.ndarray,
        right_token_uv: np.ndarray,
        valid: bool,
        grid_h: int,
        grid_w: int,
        radius_tokens: float,
    ) -> np.ndarray:
        mask = np.zeros((grid_h, grid_w), dtype=bool)
        if not valid:
            return mask
        yy, xx = np.meshgrid(
            np.arange(grid_h, dtype=np.float32),
            np.arange(grid_w, dtype=np.float32),
            indexing="ij",
        )
        points = np.stack([xx, yy], axis=-1)
        segment = right_token_uv - left_token_uv
        length2 = float(np.dot(segment, segment))
        if length2 <= 1e-12:
            closest = left_token_uv
        else:
            t = np.clip(np.sum((points - left_token_uv) * segment, axis=-1) / length2, 0.0, 1.0)
            closest = left_token_uv + t[..., None] * segment
        dist2 = np.sum((points - closest) ** 2, axis=-1)
        mask = dist2 <= radius_tokens ** 2
        return mask

    def _dino_eef_interaction_fields(self, rgb_image: np.ndarray, raw_observation: dict) -> dict:
        if raw_observation is None:
            raise KeyError(
                "current_eef interaction field requires raw RoboTwin observation with "
                "head-camera calibration and left/right endpose fields."
            )
        try:
            left = np.asarray(raw_observation["endpose"]["left_endpose"], dtype=np.float32)[:3]
            right = np.asarray(raw_observation["endpose"]["right_endpose"], dtype=np.float32)[:3]
        except KeyError as exc:
            raise KeyError(
                "current_eef interaction field requires raw_observation['endpose'] with "
                "'left_endpose' and 'right_endpose'."
            ) from exc

        image_h, image_w = rgb_image.shape[:2]
        grid_h = int(self.policy_model.dinov3_grid_height)
        grid_w = int(self.policy_model.dinov3_grid_width)
        num_tokens = grid_h * grid_w

        token_uv = np.zeros((2, 2), dtype=np.float32)
        valid = np.zeros((2,), dtype=np.float32)
        region_mask = np.zeros((2, num_tokens), dtype=np.float32)
        for arm_idx, point in enumerate((left, right)):
            uv, is_valid = self._project_point_to_head_camera(point, raw_observation)
            in_image = (
                is_valid
                and 0 <= uv[0] < image_w
                and 0 <= uv[1] < image_h
            )
            if in_image:
                token_uv[arm_idx, 0] = (uv[0] / image_w) * grid_w - 0.5
                token_uv[arm_idx, 1] = (uv[1] / image_h) * grid_h - 0.5
                valid[arm_idx] = 1.0
                region_mask[arm_idx] = self._disk_token_mask(
                    token_uv[arm_idx],
                    True,
                    grid_h,
                    grid_w,
                    self.dino_eef_mask_radius_tokens,
                ).reshape(num_tokens).astype(np.float32)

        pair_mask = self._segment_token_mask(
            token_uv[0],
            token_uv[1],
            bool(valid[0] and valid[1]),
            grid_h,
            grid_w,
            self.dino_pair_mask_radius_tokens,
        ).reshape(num_tokens).astype(np.float32)

        return {
            "dino_eef_uv": torch.from_numpy(token_uv.reshape(1, 1, 2, 2)).to(device=self.device),
            "dino_eef_valid": torch.from_numpy(valid.reshape(1, 1, 2)).to(device=self.device),
            "dino_eef_region_mask": torch.from_numpy(region_mask.reshape(1, 1, 2, num_tokens)).to(device=self.device),
            "dino_pair_region_mask": torch.from_numpy(pair_mask.reshape(1, 1, num_tokens)).to(device=self.device),
        }

    def add_online_latent_masks(self, obs_dict: dict, rgb_image: np.ndarray, raw_observation=None) -> None:
        if self.latent_mask_key is None:
            return
        if self.latent_mode == "pi3_eef_region":
            obs_dict[self.latent_mask_key] = self._eef_region_mask(
                rgb_image,
                raw_observation,
                complement=False,
            )
        elif self.latent_mode == "pi3_non_eef_region":
            obs_dict[self.latent_mask_key] = self._eef_region_mask(
                rgb_image,
                raw_observation,
                complement=True,
            )

    def add_online_interaction_fields(self, obs_dict: dict, rgb_image: np.ndarray, raw_observation=None) -> None:
        if not self.use_interaction_field:
            return
        if self.interaction_field_mode == "current_eef":
            obs_dict.update(self._dino_eef_interaction_fields(rgb_image, raw_observation))
        elif self.interaction_field_mode == "action_uv":
            raise NotImplementedError(
                "interaction_field.mode='action_uv' is reserved for action-conditioned "
                "A_lambda queries and is not implemented in this milestone. Use mode='current_eef'."
            )
        elif self.interaction_field_mode != "disabled":
            raise ValueError(f"Unsupported interaction_field.mode={self.interaction_field_mode!r}.")

    def get_action(self, rgb_image: np.ndarray, state: np.ndarray, raw_observation=None) -> np.ndarray:
        """
        Get action chunk based on current observation

        Args:
            rgb_image: numpy array [H, W, 3] RGB image
            state: numpy array [state_dim] proprioceptive state (joint positions, etc.)
            raw_observation: optional full RoboTwin observation for triadic key lookup

        Returns:
            actions: numpy array [n_action_steps, action_dim] action chunk
        """
        # Extract DinoV3 features
        dinov3_features = self.extract_dinov3_features(rgb_image)  # [1, N_patches, D]
        # Reshape to [1, N_views, N_patches, D] - single view
        dinov3_features = dinov3_features.unsqueeze(1)  # [1, 1, N_patches, D]

        # Extract Pi3 features (if enabled)
        pi3_features = None
        if self.use_pi3 and self.pi3_model is not None:
            pi3_features = self.extract_pi3_features(rgb_image)  # [1, 1, N_patches, 1024]

        # Preprocess state: [state_dim] -> tensor
        state_tensor = torch.from_numpy(state).float().to(self.device)  # [state_dim]
        state_tensor = state_tensor.unsqueeze(0)  # [1, state_dim]

        # Create observation dict with pre-extracted features
        obs_dict = {
            "dinov3_features": dinov3_features,
            "agent_pos": state_tensor,
        }

        if pi3_features is not None:
            obs_dict["pi3_features"] = pi3_features

        self.add_online_latent_masks(obs_dict, rgb_image, raw_observation=raw_observation)
        self.add_online_interaction_fields(obs_dict, rgb_image, raw_observation=raw_observation)

        if self.use_triadic_token:
            triadic_source = {"agent_pos": state}
            if raw_observation is not None:
                triadic_source["raw_observation"] = raw_observation
            triadic_state = build_triadic_state(
                triadic_source,
                self.triadic_config,
                warn_fn=lambda msg: cprint(f"[GAP] {msg}", "yellow"),
            )
            if triadic_state is None:
                raise ValueError(
                    "Checkpoint expects a triadic_state token, but the current evaluation "
                    "observation does not expose the required candidate keys. Use a vanilla "
                    "checkpoint, pass object/EEF keys, or train with triadic_mode=proprio_only_fallback."
                )
            triadic_tensor = torch.as_tensor(triadic_state, dtype=torch.float32, device=self.device)
            if triadic_tensor.ndim == 1:
                triadic_tensor = triadic_tensor.unsqueeze(0)
            elif triadic_tensor.ndim > 2:
                triadic_tensor = triadic_tensor.reshape(-1, triadic_tensor.shape[-1])[:1]
            obs_dict["triadic_state"] = triadic_tensor

        # Predict action chunk
        with torch.no_grad():
            result = self.policy_model.predict_action(obs_dict)
            # Extract action chunk [n_action_steps, action_dim]
            actions = result["action"][0].cpu().numpy()

        if self.debug:
            cprint(f"[GAP] Action predicted. Shape: {actions.shape}", "cyan")

        return actions


# ============================================================================
# RoboTwin Evaluation Interface
# ============================================================================

def encode_obs(observation):
    """
    Extract and format observation for policy input

    Args:
        observation: Raw observation from environment

    Returns:
        obs: Dict with 'rgb' and 'state'
    """
    # Extract RGB image from head camera
    rgb = observation["observation"]["head_camera"]["rgb"]  # [H, W, 3]

    # Extract robot state (joint positions)
    state = observation["joint_action"]["vector"]  # [state_dim]

    return {
        "rgb": rgb,
        "state": state,
    }


def get_model(usr_args):
    """
    Factory function to create policy for evaluation
    Required by RoboTwin evaluation framework

    Args:
        usr_args: Dict with configuration parameters
            - ckpt_path: Path to checkpoint (optional, auto-constructed if not provided)
            - task_name: Task name
            - ckpt_setting: Checkpoint setting
            - expert_data_num: Number of expert demonstrations
            - seed: Random seed
            - checkpoint_num: Checkpoint epoch number
            - device: Device to use (optional)
            - debug: Enable debug mode (optional)

    Returns:
        policy: GAPPolicyWrapper instance
    """
    # Get current directory for default paths
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Set device
    device = usr_args.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    debug = usr_args.get("debug", False)

    # Construct checkpoint path if not provided
    ckpt_path = usr_args.get("ckpt_path", None)
    if ckpt_path is None:
        # Format: checkpoints/{task_name}_{ckpt_setting}_{expert_data_num}/{checkpoint_num}.ckpt
        ckpt_dir = os.path.join(
            current_dir,
            "checkpoints",
            # f"{usr_args['task_name']}_{usr_args['ckpt_setting']}_{usr_args['expert_data_num']}"
            f"{usr_args['task_name']}_{usr_args['ckpt_setting']}_{usr_args['expert_data_num']}"
        )
        ckpt_path = os.path.join(ckpt_dir, f"{usr_args['checkpoint_num']}.ckpt")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    cprint(f"[GAP] Loading checkpoint: {ckpt_path}", "cyan")

    # Create policy
    policy = GAPPolicyWrapper(
        ckpt_path=ckpt_path,
        device=device,
        debug=debug,
    )

    return policy


def reset_model(model: GAPPolicyWrapper):
    """
    Reset model state between episodes
    Required by RoboTwin evaluation framework

    Args:
        model: GAPPolicyWrapper instance
    """
    model.reset()


def eval(TASK_ENV, model: GAPPolicyWrapper, observation, episode_info=None):
    """
    Evaluation step - execute one action chunk
    Required by RoboTwin evaluation framework

    Args:
        TASK_ENV: RoboTwin task environment
        model: GAPPolicyWrapper instance
        observation: Observation dict from environment
    """
    # Extract observation
    obs = encode_obs(observation)

    actions = model.get_action(obs["rgb"], obs["state"], raw_observation=observation)

    for action in actions:
        TASK_ENV.take_action(action)

        observation = TASK_ENV.get_obs()
        # obs = encode_obs(observation)

    return observation
