"""Record N successful demo videos of the trained policy.

Each successful pick is saved as its own mp4 file: demo_0000.mp4, demo_0001.mp4, ...
Only successful episodes are saved; failures are discarded.

Usage:
    ~/IsaacLab/isaaclab.sh -p ~/conveyor_belt/conveyor_belt/record_video.py \
        --checkpoint ~/conveyor_belt/logs/conveyor_belt/2026-04-14_17-38-33/model_499.pt \
        --num_demos 10 --enable_cameras
"""

from __future__ import annotations

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Record successful demo videos.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model .pt file.")
parser.add_argument("--num_demos", type=int, default=10, help="Number of successful demos to collect.")
parser.add_argument(
    "--output_dir", type=str,
    default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "demos"),
    help="Output directory.",
)
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner
from isaaclab.sensors import TiledCameraCfg
from isaaclab.sim.spawners.sensors import PinholeCameraCfg as PinholeCameraSpawnerCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import conveyor_belt  # noqa: F401
from conveyor_belt.agents.rsl_rl_ppo_cfg import ConveyorBeltPPORunnerCfg
from conveyor_belt.conveyor_env_cfg import ConveyorBeltEnvCfg


def log(msg):
    import sys
    print(msg, flush=True)
    sys.stdout.flush()


EYE = (1.5, -1.5, 1.0)
TARGET = (0.5, 0.0, 0.0)
WIDTH, HEIGHT = 1280, 720


def _look_at_quat_xyzw(eye, target, world_up=(0, 0, 1)):
    """Compute (x,y,z,w) quaternion for a camera looking from *eye* at *target* (world convention)."""
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    world_up = np.asarray(world_up, dtype=np.float64)

    forward = target - eye
    forward /= np.linalg.norm(forward)

    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)

    up = np.cross(right, forward)

    R = np.column_stack([forward, -right, up])

    from scipy.spatial.transform import Rotation
    return tuple(Rotation.from_matrix(R).as_quat())


def main() -> None:
    import imageio.v3 as iio

    env_cfg = ConveyorBeltEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.seed = 42

    quat = _look_at_quat_xyzw(EYE, TARGET)

    env_cfg.scene.record_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/RecordCamera",
        offset=TiledCameraCfg.OffsetCfg(pos=EYE, rot=quat, convention="world"),
        spawn=PinholeCameraSpawnerCfg(focal_length=24.0, horizontal_aperture=20.955),
        data_types=["rgb"],
        width=WIDTH,
        height=HEIGHT,
        update_period=0.0,
    )

    os.makedirs(args_cli.output_dir, exist_ok=True)

    env = gym.make("Isaac-ConveyorBelt-Franka-v0", cfg=env_cfg)
    env.reset()

    sim = env.unwrapped.sim
    camera = env.unwrapped.scene["record_camera"]

    for _ in range(10):
        sim.render()
    camera.update(sim.get_physics_dt())

    # --- Set up policy ---
    env_wrapped = RslRlVecEnvWrapper(env)

    agent_cfg = ConveyorBeltPPORunnerCfg()
    _DEPRECATED = {"stochastic", "init_noise_std", "noise_std_type", "state_dependent_std"}
    cfg_dict = agent_cfg.to_dict()
    for key in ("actor", "critic"):
        if key in cfg_dict and isinstance(cfg_dict[key], dict):
            for k in _DEPRECATED:
                cfg_dict[key].pop(k, None)

    runner = OnPolicyRunner(env_wrapped, cfg_dict, log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=agent_cfg.device)

    # --- Collect demos ---
    obs = env_wrapped.get_observations()
    term_mgr = env.unwrapped.termination_manager
    frames: list[np.ndarray] = []
    saved = 0
    target = args_cli.num_demos

    log(f"Collecting {target} successful demos...")

    while saved < target:
        with torch.no_grad():
            actions = policy(obs)
        obs, _, dones, _ = env_wrapped.step(actions)

        camera.update(sim.get_physics_dt())
        rgb = camera.data.output.get("rgb")

        if dones[0]:
            # Don't include this frame — the env already reset so the
            # camera shows the post-reset state, not the final success.
            if term_mgr.get_term("success")[0] and frames:
                fname = f"demo_{saved:04d}.mp4"
                path = os.path.join(args_cli.output_dir, fname)
                iio.imwrite(path, np.stack(frames), fps=30, codec="h264")
                saved += 1
                log(f"  [{saved}/{target}] {fname} ({len(frames)} frames)")
            else:
                log(f"  Episode failed, discarding {len(frames)} frames...")

            frames.clear()
        elif rgb is not None and rgb.numel() > 0:
            frames.append(rgb[0].cpu().numpy().astype(np.uint8))

    env_wrapped.close()
    log(f"\nDone! {saved} demos saved to {args_cli.output_dir}")


if __name__ == "__main__":
    main()
    simulation_app.close()
