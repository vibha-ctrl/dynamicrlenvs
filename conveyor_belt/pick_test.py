"""Hardcoded pick-feasibility test for each YCB object.

Runs three sequential pick trials (one per YCB object) with the belt
stopped and objects at their nominal spawn positions.  The Franka arm
executes a scripted  home → pregrasp → lower → grasp → lift  sequence
driven by joint-position waypoints.  The script reports whether each
object was successfully lifted to 20 cm above ground.

This is NOT a trained policy — it is purely kinematic feasibility
verification so you can confirm the gripper geometry/size is compatible
with each object before investing in training.

Object world positions (nominal, belt stopped):
    sugar_box      (0.50,  0.35,  ~0.069 m)
    soup_can       (0.45,  0.15,  ~0.045 m)
    mustard_bottle (0.55, -0.05,  ~0.074 m)

Robot base is at (0.05, 0.0, -0.12) in world frame, so in the robot
base frame these are:
    sugar_box      (0.45,  0.35,  ~0.189 m)
    soup_can       (0.40,  0.15,  ~0.165 m)
    mustard_bottle (0.50, -0.05,  ~0.194 m)

Usage
-----
# Recommended: open viewer so you can watch and tune:
/media/db4/wangyx/vibha/IsaacLab/isaaclab.sh -p conveyor_belt/pick_test.py --viz viser

# Headless (just print pass / fail):
/media/db4/wangyx/vibha/IsaacLab/isaaclab.sh -p conveyor_belt/pick_test.py --headless

Tuning
------
If the gripper misses an object, edit JOINT_WAYPOINTS for that object.

Key sensitivities (Franka Panda, near-default config):
  j1  →  rotates base L/R; increase to move EE left (+Y direction)
  j4  →  elbow flex; more negative raises EE, less negative lowers EE
          roughly ±0.3 m per radian at nominal reach
  j6  →  wrist; adjust to keep EE level when j4 changes
  j2  →  shoulder; increase (less negative) to extend reach forward
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Hardcoded pick-feasibility test.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# Post-launch imports
# ---------------------------------------------------------------------------
import torch  # noqa: E402
import warp as wp  # noqa: E402
import gymnasium as gym  # noqa: E402

import conveyor_belt  # noqa: E402, F401
from conveyor_belt.conveyor_env_cfg import ConveyorBeltEnvCfg  # noqa: E402

# ---------------------------------------------------------------------------
# Timing (env steps; 1 step = decimation × dt = 8 × 0.0025 s = 0.02 s)
# ---------------------------------------------------------------------------

T_SETTLE   = 60   # 1.2 s  — arm settles at home after reset
T_PREGRASP = 160  # 3.2 s  — move to position above target object
T_LOWER    = 140  # 2.8 s  — lower gripper to object height (slower = less lateral drift)
T_STABILIZE = 40  # 0.8 s  — pause at grasp height so arm/object stop moving before close
T_CLOSE    = 80   # 1.6 s  — close fingers around object
T_PRELIFT  = 80   # 1.6 s  — raise straight up (keeps grip orientation, prevents arc-induced slip)
T_LIFT     = 120  # 2.4 s  — move to final lifted pose

# Object must reach this world-z height to count as a successful pick.
SUCCESS_Z_WORLD = 0.20  # 20 cm above ground ≈ 15 cm above belt surface

# ---------------------------------------------------------------------------
# Franka Panda action-space constants
# ---------------------------------------------------------------------------

# JointPositionActionCfg(scale=0.5, use_default_offset=True) maps:
#   applied_joint_pos = default_joint_pos + action * 0.5
# → action = (target_joint_pos − default_joint_pos) / 0.5

_DEFAULTS = torch.tensor([0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741])
_SCALE = 0.5

# BinaryJointPositionActionCfg: +1.0 = open (0.04 m),  −1.0 = close (0.0 m)

# ---------------------------------------------------------------------------
# Joint-position waypoints  [j1 .. j7]
# ---------------------------------------------------------------------------
# NOTE — these are approximate estimates based on Franka Panda kinematics.
# Run with --viz viser to watch the arm and tune if needed.
#
# All positions in robot-base frame:
#   sugar_box        target EE grasp height ≈ 0.18 m
#   soup_can         target EE grasp height ≈ 0.16 m
#   mustard_bottle   target EE grasp height ≈ 0.18 m

JOINT_WAYPOINTS: dict[str, dict[str, list[float]]] = {
    # ── sugar_box  (robot frame: x=0.45, y=0.35) ──────────────────────────
    "sugar_box": {
        #                  j1     j2     j3     j4     j5     j6     j7
        "home":     [ 0.00, -0.569,  0.00, -2.810,  0.00,  3.037,  0.741],
        "pregrasp": [ 0.66, -0.20,   0.00, -2.10,   0.00,  2.00,   1.44 ],  # EE ~30 cm above belt
        "grasp":    [ 0.66,  0.10,   0.00, -2.60,   0.00,  2.58,   1.44 ],  # EE at object height
        "lift":     [ 0.66, -0.80,   0.00, -1.80,   0.00,  1.10,   1.44 ],  # EE ~45 cm above belt
    },
    # ── soup_can  (robot frame: x=0.40, y=0.15) ───────────────────────────
    "soup_can": {
        "home":     [ 0.00, -0.569,  0.00, -2.810,  0.00,  3.037,  0.741],
        "pregrasp": [ 0.36, -0.40,   0.00, -2.10,   0.00,  1.80,   1.15 ],  # higher above can
        "grasp":    [ 0.36, -0.05,   0.00, -2.20,   0.00,  2.00,   1.15 ],  # extend further to center can in fingers
        "prelift":  [ 0.36, -0.45,   0.00, -2.00,   0.00,  1.70,   1.15 ],  # straight-up raise, same lateral pose
        "lift":     [ 0.36, -0.80,   0.00, -1.80,   0.00,  1.10,   1.15 ],  # final raised position
    },
    # ── mustard_bottle  (robot frame: x=0.50, y=−0.05) ────────────────────
    "mustard_bottle": {
        "home":     [ 0.00, -0.569,  0.00, -2.810,  0.00,  3.037,  0.741],
        "pregrasp": [-0.10, -0.20,   0.00, -2.10,   0.00,  2.00,   0.69 ],
        "grasp":    [-0.10,  0.10,   0.00, -2.60,   0.00,  2.58,   0.69 ],
        "lift":     [-0.10, -0.80,   0.00, -1.80,   0.00,  1.10,   0.69 ],
    },
}

# Each trial tests one object; map name → scene asset key + starting z
OBJECTS = [
    # {"name": "sugar_box",      "scene_key": "object_0"},
    {"name": "soup_can",       "scene_key": "object_1"},
    # {"name": "mustard_bottle", "scene_key": "object_2"},
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_action(joints: list[float], open_gripper: bool, device: str) -> torch.Tensor:
    """Return a (1, 8) action tensor for the given joint target + gripper state."""
    target = torch.tensor(joints, dtype=torch.float32, device=device)
    arm = (target - _DEFAULTS.to(device)) / _SCALE          # shape (7,)
    gripper = torch.tensor([1.0 if open_gripper else -1.0], device=device)
    return torch.cat([arm, gripper]).unsqueeze(0)            # shape (1, 8)


def _step_for(env, action: torch.Tensor, n_steps: int) -> bool:
    """Step env for n_steps holding the same action.

    Returns True immediately if the episode terminates early (so the caller
    can stop driving the arm and record the result).
    """
    for _ in range(n_steps):
        _, _, terminated, truncated, _ = env.step(action)
        if terminated[0].item() or truncated[0].item():
            return True
    return False


def _object_z(env, scene_key: str) -> float:
    return wp.to_torch(env.unwrapped.scene[scene_key].data.root_pos_w)[0, 2].item()


def _ee_pos(env) -> tuple[float, float, float]:
    """Return current EE position in world frame (for diagnostic prints)."""
    ee = env.unwrapped.scene["ee_frame"]
    pos = wp.to_torch(ee.data.target_pos_w)[0, 0]  # (3,) world frame, env 0, target 0
    return pos[0].item(), pos[1].item(), pos[2].item()


# ---------------------------------------------------------------------------
# Single-object trial
# ---------------------------------------------------------------------------


def run_trial(env, obj: dict, device: str) -> dict:
    """Execute one scripted pick sequence and return a result dict."""
    name  = obj["name"]
    sk    = obj["scene_key"]
    waypts = JOINT_WAYPOINTS[name]

    print(f"\n  ── {name} ────────────────────────────────────────────────")

    env.reset()

    # ---- phase 0: home (let physics settle) ----
    print("     [settle ]  waiting for scene to settle …")
    _step_for(env, _make_action(waypts["home"], open_gripper=True, device=device), T_SETTLE)
    x, y, z = _ee_pos(env)
    print(f"               EE world pos  ({x:.3f}, {y:.3f}, {z:.3f})")

    # ---- phase 1: pre-grasp (open gripper, arm above object) ----
    print("     [pregrasp]  moving above object …")
    _step_for(env, _make_action(waypts["pregrasp"], open_gripper=True, device=device), T_PREGRASP)
    x, y, z = _ee_pos(env)
    print(f"               EE world pos  ({x:.3f}, {y:.3f}, {z:.3f})")
    print(f"               object z      {_object_z(env, sk):.3f} m")

    # ---- phase 2: lower to grasp height ----
    print("     [lower  ]  lowering to object …")
    _step_for(env, _make_action(waypts["grasp"], open_gripper=True, device=device), T_LOWER)
    x, y, z = _ee_pos(env)
    print(f"               EE world pos  ({x:.3f}, {y:.3f}, {z:.3f})")
    print(f"               object z      {_object_z(env, sk):.3f} m")

    # ---- phase 2b: stabilize (hold position, gripper open, let arm/object settle) ----
    print("     [stabilize]  holding at grasp height …")
    _step_for(env, _make_action(waypts["grasp"], open_gripper=True, device=device), T_STABILIZE)
    x, y, z = _ee_pos(env)
    print(f"               EE world pos  ({x:.3f}, {y:.3f}, {z:.3f})")

    # ---- phase 3: close gripper ----
    print("     [close  ]  closing gripper …")
    _step_for(env, _make_action(waypts["grasp"], open_gripper=False, device=device), T_CLOSE)

    # ---- phase 4a: prelift (straight up, keep grip orientation stable) ----
    if "prelift" in waypts:
        print("     [prelift]  raising straight up …")
        _step_for(env, _make_action(waypts["prelift"], open_gripper=False, device=device), T_PRELIFT)
        x, y, z = _ee_pos(env)
        print(f"               EE world pos  ({x:.3f}, {y:.3f}, {z:.3f})")
        print(f"               object z      {_object_z(env, sk):.3f} m")

    # ---- phase 4b: lift to final pose ----
    print("     [lift   ]  lifting arm …")
    _step_for(env, _make_action(waypts["lift"], open_gripper=False, device=device), T_LIFT)

    final_z = _object_z(env, sk)
    success = final_z > SUCCESS_Z_WORLD
    status  = "SUCCESS ✓" if success else "FAILED  ✗"
    print(f"     [result ]  object z = {final_z:.3f} m  →  {status}")

    return {"name": name, "success": success, "final_z": final_z}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    env_cfg = ConveyorBeltEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.seed = 0
    env_cfg.episode_length_s = 60.0  # long episode; no timeout mid-sequence

    # Stop the belt so objects sit still.
    env_cfg.events.reset_belt.params["speed_range"] = (0.0, 0.0)

    # Remove spawn randomisation so objects land at their nominal positions.
    for attr in ("reset_object_0", "reset_object_1", "reset_object_2"):
        getattr(env_cfg.events, attr).params["pose_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
        }

    env = gym.make("Isaac-ConveyorBelt-Franka-v0", cfg=env_cfg)
    device = env.unwrapped.device

    print("\n" + "=" * 60)
    print("  PICK FEASIBILITY TEST")
    print("  Belt stopped · Objects at nominal spawn positions")
    print("=" * 60)

    results = [run_trial(env, obj, device) for obj in OBJECTS]

    # ---- summary ----
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    all_pass = True
    for r in results:
        tag = "SUCCESS ✓" if r["success"] else "FAILED  ✗"
        print(f"  {r['name']:<20}  {tag}   (final z = {r['final_z']:.3f} m)")
        if not r["success"]:
            all_pass = False

    print()
    if all_pass:
        print("  All objects are physically graspable — good to train!")
    else:
        print("  Some grasps missed.  Tune JOINT_WAYPOINTS and re-run.")
        print("  Tip: run with --viz viser to watch the arm and see where it ends up.")
    print()

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
