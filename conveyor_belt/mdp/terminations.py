# Copyright (c) 2024-2026. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Custom termination terms for the conveyor-belt manipulation task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import FrameTransformer


def any_object_out_of_bounds(
    env: ManagerBasedRLEnv,
    object_names: list[str],
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    z_min: float,
) -> torch.Tensor:
    """Terminate if *any* object leaves the workspace bounding box (local frame)."""
    out = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    for name in object_names:
        obj: RigidObject = env.scene[name]
        local_pos = wp.to_torch(obj.data.root_pos_w)[:, :3] - env.scene.env_origins
        out |= (local_pos[:, 0] < x_bounds[0]) | (local_pos[:, 0] > x_bounds[1])
        out |= (local_pos[:, 1] < y_bounds[0]) | (local_pos[:, 1] > y_bounds[1])
        out |= local_pos[:, 2] < z_min
    return out


def successful_grasp_lift(
    env: ManagerBasedRLEnv,
    target_height: float,
    max_distance: float,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    object_names: list[str] = ["object_0", "object_1", "object_2"],
) -> torch.Tensor:
    """Terminate (success) when any object is lifted to ``target_height`` near the EE."""
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_pos_w = wp.to_torch(ee_frame.data.target_pos_w)[..., 0, :]

    success = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    for name in object_names:
        obj: RigidObject = env.scene[name]
        obj_pos_w = wp.to_torch(obj.data.root_pos_w)[:, :3]
        dist = torch.norm(obj_pos_w - ee_pos_w, dim=-1)
        local_z = obj_pos_w[:, 2] - env.scene.env_origins[:, 2]
        success |= (local_z > target_height) & (dist < max_distance)
    return success
