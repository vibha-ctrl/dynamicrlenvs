"""Standalone training script – PPO on the conveyor-belt pick task.

Launch
------
.. code-block:: bash

    # Standard (uses Isaac Lab's Python)
    ~/IsaacLab/isaaclab.sh -p ~/conveyor_belt/train.py --num_envs 256

    # Full-scale on RTX 5090
    ~/IsaacLab/isaaclab.sh -p ~/conveyor_belt/train.py --num_envs 4096 --max_iterations 3000
"""

from __future__ import annotations

import argparse
import os
import sys

# ---------------------------------------------------------------------------
# 1. Parse CLI & launch Omniverse  (MUST happen before any isaaclab imports)
# ---------------------------------------------------------------------------
from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="Train PPO on Conveyor Belt task.")
parser.add_argument("--num_envs", type=int, default=256, help="Number of parallel environments.")
parser.add_argument("--max_iterations", type=int, default=1500, help="PPO training iterations.")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument("--video", action="store_true", help="Record evaluation videos.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# 2. Post-launch imports (Omniverse is now running)
# ---------------------------------------------------------------------------
import time  # noqa: E402
from datetime import datetime  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

import conveyor_belt  # noqa: E402, F401  – triggers gym.register()
from conveyor_belt.agents.rsl_rl_ppo_cfg import ConveyorBeltPPORunnerCfg  # noqa: E402
from conveyor_belt.conveyor_env_cfg import ConveyorBeltEnvCfg  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# ---------------------------------------------------------------------------
# 3. Main
# ---------------------------------------------------------------------------
def main() -> None:
    # ---- environment config ----
    env_cfg = ConveyorBeltEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed

    # ---- agent config ----
    agent_cfg = ConveyorBeltPPORunnerCfg()
    agent_cfg.max_iterations = args_cli.max_iterations
    agent_cfg.seed = args_cli.seed

    # ---- logging ----
    log_root = os.path.join("logs", "conveyor_belt")
    log_dir = os.path.join(log_root, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(log_dir, exist_ok=True)
    print(f"[INFO] Logging experiment to: {log_dir}")

    # ---- create env ----
    env = gym.make(
        "Isaac-ConveyorBelt-Franka-v0",
        cfg=env_cfg,
        render_mode="rgb_array" if args_cli.video else None,
    )

    if args_cli.video:
        video_dir = os.path.join(log_dir, "videos")
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=video_dir,
            step_trigger=lambda step: step % 2000 == 0,
            video_length=200,
            disable_logger=True,
        )

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # ---- PPO runner ----
    # rsl-rl v5 MLPModel doesn't accept deprecated fields that configclass
    # serialises from RslRlMLPModelCfg.  Strip them before passing the dict.
    _DEPRECATED_MODEL_KEYS = {"stochastic", "init_noise_std", "noise_std_type", "state_dependent_std"}

    cfg_dict = agent_cfg.to_dict()
    for model_key in ("actor", "critic"):
        if model_key in cfg_dict and isinstance(cfg_dict[model_key], dict):
            for k in _DEPRECATED_MODEL_KEYS:
                cfg_dict[model_key].pop(k, None)

    runner = OnPolicyRunner(
        env, cfg_dict, log_dir=log_dir, device=agent_cfg.device
    )

    # ---- train ----
    start_time = time.time()
    runner.learn(
        num_learning_iterations=agent_cfg.max_iterations,
        init_at_random_ep_len=True,
    )
    elapsed = time.time() - start_time
    print(f"[INFO] Training completed in {elapsed:.1f}s")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
