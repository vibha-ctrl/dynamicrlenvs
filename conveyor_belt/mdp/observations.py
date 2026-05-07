"""Custom observation terms for the conveyor-belt manipulation task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import FrameTransformer


def object_positions_in_robot_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_names: list[str] = ["object_0", "object_1", "object_2"],
) -> torch.Tensor:
    """Positions of all conveyor objects expressed in the robot root frame.

    Returns shape ``(num_envs, 3 * len(object_names))``.
    """
    robot: RigidObject = env.scene[robot_cfg.name]
    root_pos_w = wp.to_torch(robot.data.root_pos_w)
    root_quat_w = wp.to_torch(robot.data.root_quat_w)
    parts: list[torch.Tensor] = []
    for name in object_names:
        obj: RigidObject = env.scene[name]
        obj_pos_w = wp.to_torch(obj.data.root_pos_w)[:, :3]
        obj_pos_b, _ = subtract_frame_transforms(root_pos_w, root_quat_w, obj_pos_w)
        parts.append(obj_pos_b)
    return torch.cat(parts, dim=-1)


def object_linear_velocities(
    env: ManagerBasedRLEnv,
    object_names: list[str] = ["object_0", "object_1", "object_2"],
) -> torch.Tensor:
    """World-frame linear velocities of all conveyor objects.

    Returns shape ``(num_envs, 3 * len(object_names))``.
    """
    parts: list[torch.Tensor] = []
    for name in object_names:
        obj: RigidObject = env.scene[name]
        parts.append(wp.to_torch(obj.data.root_lin_vel_w))
    return torch.cat(parts, dim=-1)


def target_object_one_hot(
    env: ManagerBasedRLEnv,
    object_names: list[str] = ["object_0", "object_1", "object_2"],
) -> torch.Tensor:
    """One-hot encoding of the per-env target object index.

    Returns shape ``(num_envs, len(object_names))``.
    """
    if not hasattr(env, "_target_object_idx"):
        env._target_object_idx = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    n = len(object_names)
    one_hot = torch.zeros(env.num_envs, n, device=env.device)
    one_hot.scatter_(1, env._target_object_idx.unsqueeze(1), 1.0)
    return one_hot


def ee_to_object_vectors(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    object_names: list[str] = ["object_0", "object_1", "object_2"],
) -> torch.Tensor:
    """Vectors from end-effector to each object (world frame).

    Returns shape ``(num_envs, 3 * len(object_names))``.
    """
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_pos_w = wp.to_torch(ee_frame.data.target_pos_w)[..., 0, :]
    parts: list[torch.Tensor] = []
    for name in object_names:
        obj: RigidObject = env.scene[name]
        parts.append(wp.to_torch(obj.data.root_pos_w)[:, :3] - ee_pos_w)
    return torch.cat(parts, dim=-1)
