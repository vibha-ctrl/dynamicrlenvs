"""Standalone training script – PPO on the conveyor-belt pick task.

Launch
------
.. code-block:: bash

    ~/IsaacLab/isaaclab.sh -p ~/conveyor_belt/conveyor_belt/train.py --num_envs 4096 --max_iterations 5000
"""

from __future__ import annotations

import argparse
import os

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="Train PPO on Conveyor Belt task.")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--max_iterations", type=int, default=1500)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--eval_interval", type=int, default=50,
                    help="Run deterministic eval every N iterations (0 to disable).")
parser.add_argument("--video", action="store_true")
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import time  # noqa: E402
from datetime import datetime  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

import conveyor_belt  # noqa: E402, F401
from conveyor_belt.agents.rsl_rl_ppo_cfg import ConveyorBeltPPORunnerCfg  # noqa: E402
from conveyor_belt.conveyor_env_cfg import ConveyorBeltEnvCfg  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def run_eval(
    env: RslRlVecEnvWrapper,
    runner: OnPolicyRunner,
    device: str,
    max_steps: int = 10000,
) -> dict:
    """Run the deterministic policy, collecting exactly one episode per env.

    Each env contributes one complete episode -- no env is counted twice.
    This gives an unbiased sample equivalent to running N sequential episodes.
    """
    runner.alg.eval_mode()
    policy = runner.alg.get_policy()
    num_envs = env.num_envs

    term_mgr = env.unwrapped.termination_manager
    success_idx = term_mgr._term_name_to_term_idx["success"]

    obs = env.get_observations().to(device)
    ep_rewards = torch.zeros(num_envs, device=device)
    counted = torch.zeros(num_envs, device=device, dtype=torch.bool)
    completed_rewards: list[float] = []
    total_successes = 0

    with torch.inference_mode():
        for _ in range(max_steps):
            actions = policy(obs)
            obs, rewards, dones, extras = env.step(actions.to(env.device))
            obs, rewards, dones = obs.to(device), rewards.to(device), dones.to(device)
            ep_rewards += rewards

            done_mask = dones.bool()
            if done_mask.any():
                new_done = done_mask & ~counted
                if new_done.any():
                    completed_rewards.extend(ep_rewards[new_done].tolist())
                    total_successes += (term_mgr._term_dones[:, success_idx] & new_done).sum().item()
                    counted |= new_done
                ep_rewards[done_mask] = 0.0

                if counted.all():
                    break

    rewards_t = torch.tensor(completed_rewards) if completed_rewards else torch.zeros(1)
    n = len(completed_rewards)

    runner.alg.train_mode()
    return {
        "mean_reward": rewards_t.mean().item(),
        "std_reward": rewards_t.std().item() if n > 1 else 0.0,
        "success_rate": total_successes / max(n, 1),
        "num_episodes": n,
    }


def main() -> None:
    env_cfg = ConveyorBeltEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed

    agent_cfg = ConveyorBeltPPORunnerCfg()
    agent_cfg.max_iterations = args_cli.max_iterations
    agent_cfg.seed = args_cli.seed

    log_root = os.path.join("logs", "conveyor_belt")
    log_dir = os.path.join(log_root, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(log_dir, exist_ok=True)
    print(f"[INFO] Logging experiment to: {log_dir}")

    env = gym.make(
        "Isaac-ConveyorBelt-Franka-v0",
        cfg=env_cfg,
        render_mode="rgb_array" if args_cli.video else None,
    )

    if args_cli.video:
        video_dir = os.path.join(log_dir, "videos")
        env = gym.wrappers.RecordVideo(
            env, video_folder=video_dir,
            step_trigger=lambda step: step % 2000 == 0,
            video_length=200, disable_logger=True,
        )

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    _DEPRECATED = {"stochastic", "init_noise_std", "noise_std_type", "state_dependent_std"}
    cfg_dict = agent_cfg.to_dict()
    for model_key in ("actor", "critic"):
        if model_key in cfg_dict and isinstance(cfg_dict[model_key], dict):
            for k in _DEPRECATED:
                cfg_dict[model_key].pop(k, None)

    runner = OnPolicyRunner(env, cfg_dict, log_dir=log_dir, device=agent_cfg.device)
    device = runner.device

    # --- Custom training loop with periodic deterministic evaluation ---
    env.episode_length_buf = torch.randint_like(
        env.episode_length_buf, high=int(env.max_episode_length)
    )

    obs = env.get_observations().to(device)
    runner.alg.train_mode()
    runner.logger.init_logging_writer()

    start_it = runner.current_learning_iteration
    total_it = start_it + args_cli.max_iterations
    start_time = time.time()

    for it in range(start_it, total_it):
        iter_start = time.time()

        with torch.inference_mode():
            for _ in range(runner.cfg["num_steps_per_env"]):
                actions = runner.alg.act(obs)
                obs, rewards, dones, extras = env.step(actions.to(env.device))
                obs, rewards, dones = obs.to(device), rewards.to(device), dones.to(device)
                runner.alg.process_env_step(obs, rewards, dones, extras)
                runner.logger.process_env_step(rewards, dones, extras, None)

            collect_time = time.time() - iter_start
            learn_start = time.time()
            runner.alg.compute_returns(obs)

        loss_dict = runner.alg.update()
        learn_time = time.time() - learn_start
        runner.current_learning_iteration = it

        runner.logger.log(
            it=it, start_it=start_it, total_it=total_it,
            collect_time=collect_time, learn_time=learn_time,
            loss_dict=loss_dict, learning_rate=runner.alg.learning_rate,
            action_std=runner.alg.get_policy().output_std,
            rnd_weight=None,
        )

        if runner.logger.writer is not None and it % runner.cfg["save_interval"] == 0:
            runner.save(os.path.join(log_dir, f"model_{it}.pt"))

        if args_cli.eval_interval > 0 and (it + 1) % args_cli.eval_interval == 0:
            ev = run_eval(env, runner, device)
            print(f"\n{'=' * 60}")
            print(f"  [EVAL @ iter {it + 1}]  {ev['num_episodes']} episodes")
            print(f"  mean reward: {ev['mean_reward']:.2f} +/- {ev['std_reward']:.2f}")
            print(f"  success rate: {ev['success_rate']:.1%}")
            print(f"{'=' * 60}\n")
            if runner.logger.writer is not None:
                runner.logger.writer.add_scalar("Eval/mean_reward", ev["mean_reward"], it)
                runner.logger.writer.add_scalar("Eval/std_reward", ev["std_reward"], it)
                runner.logger.writer.add_scalar("Eval/success_rate", ev["success_rate"], it)
            obs = env.get_observations().to(device)

    if runner.logger.writer is not None:
        runner.save(os.path.join(log_dir, f"model_{runner.current_learning_iteration}.pt"))
        runner.logger.stop_logging_writer()

    elapsed = time.time() - start_time
    print(f"[INFO] Training completed in {elapsed:.1f}s")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
