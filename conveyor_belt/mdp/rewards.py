"""Custom reward terms for the conveyor-belt manipulation task.

All rewards are either one-time events or progress-based.  No reward is
given for *being* in a state — only for *reaching* it or *improving*.
This prevents the agent from camping at an intermediate height.

Each env has a randomly assigned **target object** (set by the
``randomise_target_object`` event).  Only the target object is considered
for approach / grasp / lift / success rewards, forcing the policy to
learn to pick *any* of the 3 objects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply

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


def _target_obj_state(
    env: ManagerBasedRLEnv,
    object_names: list[str],
) -> torch.Tensor:
    """Return ``(num_envs, 3)`` world-frame position of each env's target object."""
    idx = env._target_object_idx  # (num_envs,) long
    all_pos = torch.stack(
        [wp.to_torch(env.scene[name].data.root_pos_w)[:, :3] for name in object_names],
        dim=1,
    )  # (num_envs, num_objects, 3)
    return all_pos[torch.arange(env.num_envs, device=env.device), idx]  # (num_envs, 3)


def _target_grasped_height(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg,
    object_names: list[str],
    max_dist: float = 0.15,
) -> torch.Tensor:
    """Height of the target object if it is within *max_dist* of the gripper, else 0."""
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_pos_w = wp.to_torch(ee_frame.data.target_pos_w)[..., 0, :]
    obj_pos_w = _target_obj_state(env, object_names)

    dist = torch.norm(obj_pos_w - ee_pos_w, dim=-1)
    local_z = obj_pos_w[:, 2] - env.scene.env_origins[:, 2]
    near = (dist < max_dist).float()
    return local_z * near


# ── 1. approach (target object only) ─────────────────────────────────────

def approach_object(
    env: ManagerBasedRLEnv,
    std: float,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    object_names: list[str] = ["object_0", "object_1", "object_2"],
) -> torch.Tensor:
    """Tanh-kernel approach reward toward the target object."""
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_pos_w = wp.to_torch(ee_frame.data.target_pos_w)[..., 0, :]
    obj_pos_w = _target_obj_state(env, object_names)

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
    obj_pos_w = _target_obj_state(env, object_names)

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
    """Reward only for *new* upward progress of the target object."""
    if not hasattr(env, "_lift_best_height"):
        env._lift_best_height = torch.zeros(env.num_envs, device=env.device)

    env._lift_best_height[_detect_resets(env, "lift")] = 0.0

    current = _target_grasped_height(env, ee_frame_cfg, object_names)
    progress = torch.clamp(current - env._lift_best_height, min=0.0)
    env._lift_best_height = torch.maximum(env._lift_best_height, current)
    return progress


# ── 4. height milestones (one-time threshold bonuses) ────────────────────

def height_milestones(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    object_names: list[str] = ["object_0", "object_1", "object_2"],
) -> torch.Tensor:
    """One-time bonuses at 8 / 12 / 16 cm while target object is near gripper."""
    thresholds = torch.tensor([0.08, 0.12, 0.16], device=env.device)
    bonuses = torch.tensor([10.0, 20.0, 30.0], device=env.device)
    n = len(thresholds)

    if not hasattr(env, "_milestones_hit"):
        env._milestones_hit = torch.zeros(env.num_envs, n, device=env.device, dtype=torch.bool)

    env._milestones_hit[_detect_resets(env, "milestone")] = False

    best_h = _target_grasped_height(env, ee_frame_cfg, object_names)

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
    obj_pos_w = _target_obj_state(env, object_names)

    dist = torch.norm(obj_pos_w - ee_pos_w, dim=-1)
    local_z = obj_pos_w[:, 2] - env.scene.env_origins[:, 2]
    success = (local_z > target_height) & (dist < max_grasp_distance)
    return success.float()


# ── 6. time penalty ─────────────────────────────────────────────────────

def alive_cost(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Constant 1.0 per step.  Use a negative weight (e.g. -0.01)."""
    return torch.ones(env.num_envs, device=env.device)


# ── 7. gripper shaping ───────────────────────────────────────────────────

def _gripper_close_cmd(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return bool mask (num_envs,) for whether the gripper was commanded to close.

    BinaryJointPositionAction treats raw_action > 0 as "close".
    """
    term = env.action_manager.get_term("gripper_action")
    raw = term.raw_actions  # (N, 1) or (N,)
    if raw.dim() > 1:
        raw = raw[:, 0]
    return raw > 0.0


def gripper_close_near_target(
    env: ManagerBasedRLEnv,
    max_distance: float = 0.10,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    object_names: list[str] = ["object_0", "object_1", "object_2"],
) -> torch.Tensor:
    """Reward per step when the gripper is commanded closed *and* EE is near target.

    Gated on proximity so the agent can't hack the reward by simply closing
    the gripper immediately on reset.
    """
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_pos_w = wp.to_torch(ee_frame.data.target_pos_w)[..., 0, :]
    obj_pos_w = _target_obj_state(env, object_names)

    dist = torch.norm(obj_pos_w - ee_pos_w, dim=-1)
    near = dist < max_distance
    closed = _gripper_close_cmd(env)
    return (closed & near).float()


def gripper_reopen_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalise *opening* the gripper at any point after it has ever been closed.

    Sticky within an episode: once the gripper closes, every subsequent step
    in which it is commanded open contributes +1 to this term (use a
    **negative weight**). Cleared on reset.
    """
    if not hasattr(env, "_gripper_has_closed"):
        env._gripper_has_closed = torch.zeros(
            env.num_envs, device=env.device, dtype=torch.bool
        )

    env._gripper_has_closed[_detect_resets(env, "gripper_reopen")] = False

    closed_now = _gripper_close_cmd(env)
    open_now = ~closed_now

    penalty = env._gripper_has_closed & open_now
    env._gripper_has_closed |= closed_now
    return penalty.float()


# ── 8. top-down orientation reward ───────────────────────────────────────

def gripper_downward(
    env: ManagerBasedRLEnv,
    max_distance: float = 0.20,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    object_names: list[str] = ["object_0", "object_1", "object_2"],
) -> torch.Tensor:
    """Reward the gripper for pointing downward, gated on proximity to target.

    The Franka ``panda_hand`` local +Z axis points out of the gripper along
    the grasp direction.  A top-down grasp thus means the world-frame Z
    component of that axis is ``-1``.  Returns ``score * near`` where:

    - ``score = -world_z[:, 2]`` → ``+1`` when pointing straight down,
      ``0`` horizontal, ``-1`` pointing up.
    - ``near  = 1`` when EE is within ``max_distance`` of the target object,
      else ``0``.
    """
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_pos_w = wp.to_torch(ee_frame.data.target_pos_w)[..., 0, :]
    ee_quat_w = wp.to_torch(ee_frame.data.target_quat_w)[..., 0, :]  # (N, 4) xyzw

    local_z = torch.zeros(env.num_envs, 3, device=env.device)
    local_z[:, 2] = 1.0
    world_z = quat_apply(ee_quat_w, local_z)  # (N, 3)
    score = -world_z[:, 2]  # +1 straight down, -1 straight up

    obj_pos_w = _target_obj_state(env, object_names)
    dist = torch.norm(obj_pos_w - ee_pos_w, dim=-1)
    near = (dist < max_distance).float()

    return score * near
