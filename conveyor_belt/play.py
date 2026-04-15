"""Play a trained checkpoint with visualization.

Usage:
    ~/IsaacLab/isaaclab.sh -p ~/conveyor_belt/conveyor_belt/play.py --checkpoint <path> --viz viser
"""

from __future__ import annotations

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play a trained conveyor belt policy.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model .pt file.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--steps", type=int, default=10000, help="Steps to run.")
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import conveyor_belt
from conveyor_belt.agents.rsl_rl_ppo_cfg import ConveyorBeltPPORunnerCfg
from conveyor_belt.conveyor_env_cfg import ConveyorBeltEnvCfg


def main() -> None:
    env_cfg = ConveyorBeltEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = 42

    env = gym.make("Isaac-ConveyorBelt-Franka-v0", cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    agent_cfg = ConveyorBeltPPORunnerCfg()

    _DEPRECATED = {"stochastic", "init_noise_std", "noise_std_type", "state_dependent_std"}
    cfg_dict = agent_cfg.to_dict()
    for key in ("actor", "critic"):
        if key in cfg_dict and isinstance(cfg_dict[key], dict):
            for k in _DEPRECATED:
                cfg_dict[key].pop(k, None)

    runner = OnPolicyRunner(env, cfg_dict, log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint)

    policy = runner.get_inference_policy(device=agent_cfg.device)

    obs = env.get_observations()
    device = agent_cfg.device
    num_envs = args_cli.num_envs

    from collections import deque

    term_mgr = env.unwrapped.termination_manager
    success_idx = term_mgr._term_name_to_term_idx["success"]

    WINDOW = 20
    ep_rewards = torch.zeros(num_envs, device=device)
    recent_rewards = deque(maxlen=WINDOW)
    recent_successes = deque(maxlen=WINDOW)
    total_episodes = 0
    total_successes = 0
    last_print_episodes = 0

    print(f"\nRunning {args_cli.steps} steps with trained policy...\n")

    for step in range(args_cli.steps):
        with torch.no_grad():
            actions = policy(obs)
        obs, rewards, dones, infos = env.step(actions)
        ep_rewards += rewards.to(device)

        done_mask = dones.to(device).bool()
        if done_mask.any():
            n_success = term_mgr._term_dones[:, success_idx].sum().item()
            n_done = done_mask.sum().item()
            for r in ep_rewards[done_mask].tolist():
                recent_rewards.append(r)
                total_episodes += 1
            for _ in range(int(n_success)):
                recent_successes.append(1)
            for _ in range(n_done - int(n_success)):
                recent_successes.append(0)
            total_successes += int(n_success)
            ep_rewards[done_mask] = 0.0

        if total_episodes >= last_print_episodes + WINDOW and len(recent_rewards) > 0:
            mean_r = sum(recent_rewards) / len(recent_rewards)
            succ = sum(recent_successes) / len(recent_successes)
            print(
                f"  episodes: {total_episodes:>5d}  |  last {len(recent_rewards)} eps  |  "
                f"mean reward: {mean_r:.2f}  |  success: {succ:.1%}"
            )
            last_print_episodes = total_episodes

    if total_episodes > 0:
        overall_succ = total_successes / total_episodes
        print(
            f"\nDone. {total_episodes} total episodes  |  "
            f"overall success: {overall_succ:.1%}"
        )
    else:
        print("\nDone. No episodes completed.")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
