"""Custom event terms for the conveyor-belt manipulation task.

Conveyor physics
~~~~~~~~~~~~~~~~
The Isaac Sim *Conveyor Belt Utility* (``isaacsim.asset.gen.conveyor``)
works by applying a target velocity to a rigid-body prim every OmniGraph
tick.  We replicate the same mechanism inside the Isaac Lab manager-based
workflow:

* ``reset_conveyor_belt``  – sets belt pose + velocity on episode reset.
* ``drive_conveyor_belt``  – interval event that **resets pose and re-applies
  velocity** each tick, preventing the dynamic belt from drifting in space.
* ``reset_object_on_conveyor`` – randomises each object's pose and gives
  it an initial velocity matching the belt so there is no contact impulse
  at spawn.

Objects are then transported by **physics friction** with the high-mass
belt, not by having their velocities overwritten directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject
    from isaaclab.envs import ManagerBasedRLEnv


def _conveyor_velocity_tensor(
    base: tuple[float, float, float],
    noise: float,
    n: int,
    device: torch.device,
) -> torch.Tensor:
    """Build an (n, 3) linear-velocity tensor with noise on non-zero axes only.

    Noise is restricted to the axes that carry the conveyor velocity so the
    belt / objects don't acquire spurious lateral or vertical drift.
    """
    vel = torch.tensor(base, device=device).unsqueeze(0).expand(n, -1).clone()
    if noise > 0.0:
        mask = torch.tensor(
            [abs(v) > 1e-6 for v in base], device=device, dtype=torch.bool
        )
        perturbation = torch.empty(n, 3, device=device).uniform_(-noise, noise)
        vel[:, mask] += perturbation[:, mask]
    return vel


# ---------------------------------------------------------------------------
# Belt events (mirrors Isaac Sim Conveyor Belt Utility)
# ---------------------------------------------------------------------------


def reset_conveyor_belt(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    belt_cfg: SceneEntityCfg,
    conveyor_velocity: tuple[float, float, float],
    velocity_noise: float,
) -> None:
    """Reset the conveyor belt to its default pose with randomised velocity.

    Called on episode reset to restore the belt position (preventing any
    accumulated drift) and to apply the conveyor speed ± small noise.
    """
    belt: RigidObject = env.scene[belt_cfg.name]
    n = len(env_ids)

    state = wp.to_torch(belt.data.default_root_state)[env_ids].clone()
    state[:, :3] += env.scene.env_origins[env_ids]

    lin_vel = _conveyor_velocity_tensor(
        conveyor_velocity, velocity_noise, n, env.device
    )
    state[:, 7:10] = lin_vel
    state[:, 10:13] = 0.0

    belt.write_root_pose_to_sim(state[:, :7], env_ids)
    belt.write_root_velocity_to_sim(state[:, 7:], env_ids)


def drive_conveyor_belt(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    belt_cfg: SceneEntityCfg,
    conveyor_velocity: tuple[float, float, float],
    velocity_noise: float,
) -> None:
    """Maintain the belt's target velocity **and** reset its pose (interval).

    Because the belt is a dynamic rigid body, PhysX integrates its velocity
    into position each step.  At 0.08 m/s and a 0.05 s interval the belt
    drifts ~4 mm between calls — over a full 8 s episode that compounds to
    0.64 m, nearly the belt's entire length.

    We therefore **snap the belt back to its default pose** each interval
    before re-applying velocity.  The ~4 mm correction is well within PhysX
    depenetration tolerances and does not cause visible jitter for objects
    resting on the belt.

    This mirrors the Isaac Sim Conveyor Belt Utility, whose OmniGraph node
    runs every physics tick, keeping the belt effectively stationary while
    its velocity field drags objects via friction.
    """
    belt: RigidObject = env.scene[belt_cfg.name]
    n = len(env_ids)

    # --- reset pose to prevent translational drift ---
    state = wp.to_torch(belt.data.default_root_state)[env_ids].clone()
    state[:, :3] += env.scene.env_origins[env_ids]
    belt.write_root_pose_to_sim(state[:, :7], env_ids)

    # --- re-apply conveyor velocity ---
    vel = torch.zeros(n, 6, device=env.device)
    vel[:, :3] = _conveyor_velocity_tensor(
        conveyor_velocity, velocity_noise, n, env.device
    )
    belt.write_root_velocity_to_sim(vel, env_ids=env_ids)


# ---------------------------------------------------------------------------
# Object reset
# ---------------------------------------------------------------------------


def reset_object_on_conveyor(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    pose_range: dict[str, tuple[float, float]],
    yaw_range: tuple[float, float],
    conveyor_velocity: tuple[float, float, float],
    velocity_noise: float,
) -> None:
    """Reset one rigid object to a random pose on the conveyor belt.

    * Position = default init position + uniform offset per axis.
    * Orientation = random yaw (rotation about world z).
    * Linear velocity = conveyor speed ± noise (noise on travel axis only)
      so there is no impulse when the object is placed on the moving belt.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    n = len(env_ids)

    root_states = wp.to_torch(asset.data.default_root_state)[env_ids].clone()

    # --- position ---
    for i, key in enumerate(("x", "y", "z")):
        lo, hi = pose_range.get(key, (0.0, 0.0))
        root_states[:, i] += torch.empty(n, device=env.device).uniform_(lo, hi)
    root_states[:, :3] += env.scene.env_origins[env_ids]

    # --- random yaw ---
    yaw = torch.empty(n, device=env.device).uniform_(*yaw_range)
    root_states[:, 3] = torch.cos(yaw * 0.5)  # qw
    root_states[:, 4] = 0.0                    # qx
    root_states[:, 5] = 0.0                    # qy
    root_states[:, 6] = torch.sin(yaw * 0.5)  # qz

    # --- match belt velocity (noise on travel axis only) ---
    root_states[:, 7:10] = _conveyor_velocity_tensor(
        conveyor_velocity, velocity_noise, n, env.device
    )
    root_states[:, 10:13] = 0.0

    asset.write_root_pose_to_sim(root_states[:, :7], env_ids)
    asset.write_root_velocity_to_sim(root_states[:, 7:], env_ids)
