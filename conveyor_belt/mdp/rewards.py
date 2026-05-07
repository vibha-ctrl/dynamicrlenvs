"""Custom reward terms for the conveyor-belt manipulation task.

All rewards are either one-time events or progress-based.  No reward is
given for *being* in a state — only for *reaching* it or *improving*.
This prevents the agent from camping at an intermediate height.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import FrameTransformer


# ── helpers ──────────────────────────────────────────────────────────────

def _detect_resets(env: ManagerBasedRLEnv, key: str) -> torch.Tensor:
    """Return a bool mask of envs that just reset (episode_length decreased)."""
    attr = f"_reset_prev_{key}"
    if not hasattr(env, attr):
        setattr(env, attr, torch.zeros(env.num_envs, device=env.device, dtype=torch.long))
    prev = getattr(env, attr)
    just_reset = env.episode_length_buf < prev
    setattr(env, attr, env.episode_length_buf.clone())
    return just_reset


def _target_object_pos_w(env: ManagerBasedRLEnv, object_names: list[str]) -> torch.Tensor:
    """Returns (num_envs, 3) world-frame position of the target object per env."""
    if not hasattr(env, "_target_object_idx"):
        env._target_object_idx = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    positions = torch.stack(
        [wp.to_torch(env.scene[name].data.root_pos_w)[:, :3] for name in object_names], dim=1
    )  # (num_envs, num_objects, 3)
    idx = env._target_object_idx.view(-1, 1, 1).expand(-1, 1, 3)
    return positions.gather(1, idx).squeeze(1)  # (num_envs, 3)


def _best_grasped_height(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg,
    object_names: list[str],
    max_dist: float = 0.15,
) -> torch.Tensor:
    """Height of the target object if within *max_dist* of the gripper, else 0."""
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_pos_w = wp.to_torch(ee_frame.data.target_pos_w)[..., 0, :]
    obj_pos_w = _target_object_pos_w(env, object_names)
    dist = torch.norm(obj_pos_w - ee_pos_w, dim=-1)
    local_z = obj_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return local_z * (dist < max_dist).float()


# ── 1. approach (turns off after grasp) ──────────────────────────────────

def approach_object(
    env: ManagerBasedRLEnv,
    std: float,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    object_names: list[str] = ["object_0", "object_1", "object_2"],
) -> torch.Tensor:
    """Tanh-kernel approach reward toward the target object only."""
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_pos_w = wp.to_torch(ee_frame.data.target_pos_w)[..., 0, :]
    obj_pos_w = _target_object_pos_w(env, object_names)
    dist = torch.norm(obj_pos_w - ee_pos_w, dim=-1)
    return 1.0 - torch.tanh(dist / std)


# ── 2. one-time grasp event ─────────────────────────────────────────────

def grasp_event(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    max_grasp_distance: float,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    object_names: list[str] = ["object_0", "object_1", "object_2"],
) -> torch.Tensor:
    """One-time bonus when the robot first grasps and lifts the target object."""
    if not hasattr(env, "_grasp_event_given"):
        env._grasp_event_given = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

    env._grasp_event_given[_detect_resets(env, "grasp")] = False

    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_pos_w = wp.to_torch(ee_frame.data.target_pos_w)[..., 0, :]
    obj_pos_w = _target_object_pos_w(env, object_names)
    dist = torch.norm(obj_pos_w - ee_pos_w, dim=-1)
    local_z = obj_pos_w[:, 2] - env.scene.env_origins[:, 2]
    currently_grasped = (dist < max_grasp_distance) & (local_z > minimal_height)

    new_grasp = currently_grasped & ~env._grasp_event_given
    env._grasp_event_given |= currently_grasped
    return new_grasp.float()


# ── 3. lift progress (delta height) ─────────────────────────────────────

def lift_progress(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    object_names: list[str] = ["object_0", "object_1", "object_2"],
) -> torch.Tensor:
    """Reward only for *new* upward progress.  Hovering pays nothing."""
    if not hasattr(env, "_lift_best_height"):
        env._lift_best_height = torch.zeros(env.num_envs, device=env.device)

    env._lift_best_height[_detect_resets(env, "lift")] = 0.0

    current = _best_grasped_height(env, ee_frame_cfg, object_names)
    progress = torch.clamp(current - env._lift_best_height, min=0.0)
    env._lift_best_height = torch.maximum(env._lift_best_height, current)
    return progress


# ── 4. height milestones (one-time threshold bonuses) ────────────────────

def height_milestones(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    object_names: list[str] = ["object_0", "object_1", "object_2"],
) -> torch.Tensor:
    """One-time bonuses at 8 / 12 / 16 cm while object is near gripper.

    Returns the sum of newly-crossed milestone values.  Use weight=1.0
    in the config since bonus magnitudes are baked in.
    """
    thresholds = torch.tensor([0.08, 0.12, 0.16], device=env.device)
    bonuses = torch.tensor([10.0, 20.0, 30.0], device=env.device)
    n = len(thresholds)

    if not hasattr(env, "_milestones_hit"):
        env._milestones_hit = torch.zeros(env.num_envs, n, device=env.device, dtype=torch.bool)

    env._milestones_hit[_detect_resets(env, "milestone")] = False

    best_h = _best_grasped_height(env, ee_frame_cfg, object_names)

    reward = torch.zeros(env.num_envs, device=env.device)
    for i in range(n):
        newly_crossed = (best_h > thresholds[i]) & ~env._milestones_hit[:, i]
        reward += newly_crossed.float() * bonuses[i]
        env._milestones_hit[:, i] |= best_h > thresholds[i]
    return reward


# ── 5. success bonus (gated on controlled grasp) ────────────────────────

def success_bonus(
    env: ManagerBasedRLEnv,
    target_height: float,
    max_grasp_distance: float = 0.12,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    object_names: list[str] = ["object_0", "object_1", "object_2"],
) -> torch.Tensor:
    """Large reward when the target object is above target_height AND near gripper."""
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_pos_w = wp.to_torch(ee_frame.data.target_pos_w)[..., 0, :]
    obj_pos_w = _target_object_pos_w(env, object_names)
    dist = torch.norm(obj_pos_w - ee_pos_w, dim=-1)
    local_z = obj_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return ((local_z > target_height) & (dist < max_grasp_distance)).float()


# ── 6. time penalty ─────────────────────────────────────────────────────

def alive_cost(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Constant 1.0 per step.  Use a negative weight (e.g. -0.01)."""
    return torch.ones(env.num_envs, device=env.device)
