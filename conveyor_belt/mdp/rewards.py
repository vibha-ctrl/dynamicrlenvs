# Copyright (c) 2024-2026. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Custom reward terms for the conveyor-belt manipulation task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import FrameTransformer


def closest_object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    object_names: list[str] = ["object_0", "object_1", "object_2"],
) -> torch.Tensor:
    """Tanh-kernel reward for reaching the closest conveyor object."""
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_pos_w = wp.to_torch(ee_frame.data.target_pos_w)[..., 0, :]

    min_dist = torch.full((env.num_envs,), float("inf"), device=env.device)
    for name in object_names:
        obj: RigidObject = env.scene[name]
        obj_pos_w = wp.to_torch(obj.data.root_pos_w)[:, :3]
        dist = torch.norm(obj_pos_w - ee_pos_w, dim=-1)
        min_dist = torch.minimum(min_dist, dist)

    return 1.0 - torch.tanh(min_dist / std)


def any_object_lifted(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    object_names: list[str] = ["object_0", "object_1", "object_2"],
) -> torch.Tensor:
    """Binary reward: 1 if *any* object is above ``minimal_height`` (local z)."""
    lifted = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    for name in object_names:
        obj: RigidObject = env.scene[name]
        obj_pos_w = wp.to_torch(obj.data.root_pos_w)
        local_z = obj_pos_w[:, 2] - env.scene.env_origins[:, 2]
        lifted |= local_z > minimal_height
    return lifted.float()


def grasp_and_lift(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    max_grasp_distance: float,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    object_names: list[str] = ["object_0", "object_1", "object_2"],
) -> torch.Tensor:
    """Reward for simultaneously grasping (EE close) and lifting an object."""
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_pos_w = wp.to_torch(ee_frame.data.target_pos_w)[..., 0, :]

    reward = torch.zeros(env.num_envs, device=env.device)
    for name in object_names:
        obj: RigidObject = env.scene[name]
        obj_pos_w = wp.to_torch(obj.data.root_pos_w)[:, :3]
        dist = torch.norm(obj_pos_w - ee_pos_w, dim=-1)
        local_z = obj_pos_w[:, 2] - env.scene.env_origins[:, 2]
        grasped_and_lifted = (dist < max_grasp_distance) & (local_z > minimal_height)
        reward = torch.maximum(reward, grasped_and_lifted.float())
    return reward
