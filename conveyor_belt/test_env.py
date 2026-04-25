"""Quick smoke-test for the conveyor-belt environment.

Creates the env with a handful of parallel instances, runs random actions
for a few episodes, and prints diagnostics.  Useful for verifying that the
scene loads, physics runs, and the observation / reward / termination
pipeline is wired correctly — before committing to a full training run.

Usage
-----
.. code-block:: bash

    # Headless (fastest — no GUI window)
    ~/IsaacLab/isaaclab.sh -p ~/conveyor_belt/test_env.py --num_envs 4 --headless

    # With viewer (see the sim in the Omniverse viewport)
    ~/IsaacLab/isaaclab.sh -p ~/conveyor_belt/test_env.py --num_envs 4

    # With viser browser viewer (SSH-tunnel port 8080 first)
    ~/IsaacLab/isaaclab.sh -p ~/conveyor_belt/test_env.py --num_envs 4 --viz viser
"""

from __future__ import annotations

import argparse

# ---------------------------------------------------------------------------
# Parse CLI & launch sim  (MUST precede any isaaclab imports)
# ---------------------------------------------------------------------------
from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="Smoke-test the conveyor-belt env.")
parser.add_argument("--num_envs", type=int, default=4, help="Parallel environments.")
parser.add_argument("--steps", type=int, default=200, help="Env steps to run.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# Post-launch imports
# ---------------------------------------------------------------------------
import torch  # noqa: E402
import gymnasium as gym  # noqa: E402

import conveyor_belt  # noqa: E402, F401  – triggers gym.register()
from conveyor_belt.conveyor_env_cfg import ConveyorBeltEnvCfg  # noqa: E402


def main() -> None:
    # ---- configure ----
    env_cfg = ConveyorBeltEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = 42

    # ---- create ----
    print(f"\n{'='*60}")
    print(f" Creating env with {args_cli.num_envs} instances ...")
    print(f"{'='*60}\n")

    env = gym.make("Isaac-ConveyorBelt-Franka-v0", cfg=env_cfg)

    obs, info = env.reset()
    print(f"  Observation space : {env.observation_space}")
    print(f"  Action space      : {env.action_space}")
    print(f"  Obs tensor shape  : {obs['policy'].shape}")
    print()

    # ---- run random actions ----
    total_reward = torch.zeros(args_cli.num_envs, device=obs["policy"].device)
    episodes_completed = 0

    print(f"  Running {args_cli.steps} steps with random actions ...\n")

    for step in range(args_cli.steps):
        action = 2 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        done = terminated | truncated
        episodes_completed += int(done.sum().item() if isinstance(done, torch.Tensor) else sum(done))

        if (step + 1) % 50 == 0:
            print(f"    step {step+1:>4d}  |  mean cumulative reward: {total_reward.mean().item():>8.3f}"
                  f"  |  episodes done so far: {episodes_completed}")

    # ---- summary ----
    print(f"\n{'='*60}")
    print(f" Test complete")
    print(f"{'='*60}")
    print(f"  Total steps           : {args_cli.steps}")
    print(f"  Episodes completed    : {episodes_completed}")
    print(f"  Mean cumulative reward: {total_reward.mean().item():.3f}")
    print(f"  Final obs shape       : {obs['policy'].shape}")
    print()

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
