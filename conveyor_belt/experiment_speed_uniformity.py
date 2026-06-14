"""Visual belt speed sweep — open in viser and watch.

Cycles through belt speeds 0.05 → 0.50 m/s in 0.05 increments.
Objects (10 cm, 7 cm, 4 cm) are placed at the same fixed positions every
trial.  Each speed runs until ALL THREE objects have fallen off the end of
the belt, then immediately moves to the next speed.

Usage
-----
    python conveyor_belt/experiment_speed_uniformity.py --num_envs 1 --viz viser
"""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Visual belt speed sweep.")
parser.add_argument("--num_envs", type=int, default=1,
                    help="Number of parallel environments.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# Post-launch imports
# ---------------------------------------------------------------------------
import torch
import warp as wp
import gymnasium as gym

import conveyor_belt  # noqa: F401 — triggers gym.register()
from conveyor_belt.conveyor_env_cfg import ConveyorBeltEnvCfg
from conveyor_belt.scene_cfg import (
    _OBJECT_HEIGHTS,
    _object_spawn_z,
    BELT_CENTER_X,
    BELT_WIDTH,
    BELT_LENGTH,
)

BELT_SPEEDS  = [round(0.7 + 0.1 * i, 2) for i in range(14)]  # 0.7, 0.8, … 2.0 m/s
OBJECT_NAMES = ["object_0", "object_1", "object_2"]

# Object is "off the belt" when its Y (local) drops below this threshold
_FALLOFF_Y = -BELT_LENGTH / 2 - 0.05


def _apply_speed(unwrapped, scene, device: str, vy: float) -> None:
    unwrapped._belt_velocity = torch.zeros(1, 3, device=device)
    unwrapped._belt_velocity[0, 1] = vy
    belt = scene["conveyor_belt"]
    belt_vel = torch.zeros(1, 6, device=device)
    belt_vel[0, 1] = vy
    belt.write_root_velocity_to_sim(belt_vel)


def _place_objects(scene, device: str, fixed_xy: list[tuple[float, float]],
                   vy: float) -> None:
    env_origin = scene.env_origins[0]
    for i, name in enumerate(OBJECT_NAMES):
        obj = scene[name]
        h   = _OBJECT_HEIGHTS[i]
        pose = torch.tensor([[
            fixed_xy[i][0] + env_origin[0].item(),
            fixed_xy[i][1] + env_origin[1].item(),
            _object_spawn_z(h) + env_origin[2].item(),
            1.0, 0.0, 0.0, 0.0,
        ]], device=device)
        vel = torch.zeros(1, 6, device=device)
        vel[0, 1] = vy
        obj.write_root_pose_to_sim(pose)
        obj.write_root_velocity_to_sim(vel)


def _all_fallen(scene, device: str) -> bool:
    """Return True once every object has left the belt.

    An object counts as gone if it has either:
    - travelled past the far Y end of the belt, OR
    - dropped well below the belt surface (fell off the edge).
    """
    env_origin = scene.env_origins[0]
    for name in OBJECT_NAMES:
        obj = scene[name]
        pos = wp.to_torch(obj.data.root_pos_w)[0]
        y_local = pos[1].item() - env_origin[1].item()
        z_world = pos[2].item()
        past_end    = y_local < _FALLOFF_Y
        dropped_off = z_world < (env_origin[2].item() - 0.15)  # >15 cm below belt
        if not (past_end or dropped_off):
            return False
    return True


def main() -> None:
    # Fixed positions: tallest first (lowest Y = closest to robot end),
    # each 5 cm apart in Y, in separate X lanes spread across the belt.
    # object_0 = 10 cm (front), object_1 = 7 cm (middle), object_2 = 4 cm (back)
    fixed_xy = [
        (0.38, 0.70),   # tallest  — left lane
        (0.50, 0.75),   # medium   — centre lane
        (0.62, 0.80),   # shortest — right lane
    ]

    env_cfg = ConveyorBeltEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = 0
    env = gym.make("Isaac-ConveyorBelt-Franka-v0", cfg=env_cfg)
    unwrapped   = env.unwrapped
    device      = unwrapped.device
    scene       = unwrapped.scene
    zero_action = torch.zeros(env.action_space.shape, device=device)

    print(f"\nObject heights : {[f'{h*100:.0f} cm' for h in _OBJECT_HEIGHTS]}")
    print(f"Fixed spawn    : {[(f'{x:.3f}', f'{y:.3f}') for x, y in fixed_xy]}")
    print()

    # Disable all termination conditions so env.step() never auto-resets.
    # Our _all_fallen() check controls when each trial ends instead.
    tm = unwrapped.termination_manager
    tm._term_names.clear()
    tm._term_cfgs.clear()
    tm._term_name_to_term_idx.clear()
    tm._term_dones = torch.zeros((unwrapped.num_envs, 0), device=unwrapped.device, dtype=torch.bool)

    for speed in BELT_SPEEDS:
        vy = -speed
        env.reset()
        _apply_speed(unwrapped, scene, device, vy)
        _place_objects(scene, device, fixed_xy, vy)

        print(f"  belt speed: {speed:.2f} m/s  — waiting for all objects to fall off …")
        step = 0
        while not _all_fallen(scene, device):
            env.step(zero_action)
            step += 1
        print(f"    done after {step} steps")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
