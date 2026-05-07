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


def all_objects_out_of_bounds(
    env: ManagerBasedRLEnv,
    object_names: list[str],
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    z_min: float,
) -> torch.Tensor:
    """Terminate only when *all* objects have left the workspace bounding box."""
    all_out = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
    for name in object_names:
        obj: RigidObject = env.scene[name]
        local_pos = wp.to_torch(obj.data.root_pos_w)[:, :3] - env.scene.env_origins
        obj_out = (
            (local_pos[:, 0] < x_bounds[0]) | (local_pos[:, 0] > x_bounds[1])
            | (local_pos[:, 1] < y_bounds[0]) | (local_pos[:, 1] > y_bounds[1])
            | (local_pos[:, 2] < z_min)
        )
        all_out &= obj_out
    return all_out


def successful_grasp_lift(
    env: ManagerBasedRLEnv,
    target_height: float,
    max_grasp_distance: float = 0.12,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    object_names: list[str] = ["object_0", "object_1", "object_2"],
) -> torch.Tensor:
    """Terminate (success) when the target object is above target_height AND held near gripper."""
    if not hasattr(env, "_target_object_idx"):
        env._target_object_idx = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_pos_w = wp.to_torch(ee_frame.data.target_pos_w)[..., 0, :]

    positions = torch.stack(
        [wp.to_torch(env.scene[name].data.root_pos_w)[:, :3] for name in object_names], dim=1
    )
    idx = env._target_object_idx.view(-1, 1, 1).expand(-1, 1, 3)
    obj_pos_w = positions.gather(1, idx).squeeze(1)

    dist = torch.norm(obj_pos_w - ee_pos_w, dim=-1)
    local_z = obj_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return (local_z > target_height) & (dist < max_grasp_distance)
