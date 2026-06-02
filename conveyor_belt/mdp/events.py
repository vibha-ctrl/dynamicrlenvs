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
    speed_range: tuple[float, float] | None = None,
    randomize_direction: bool = False,
) -> None:
    """Reset the conveyor belt to its default pose with randomised velocity.

    If ``speed_range`` is provided, each env gets a random speed in that range
    instead of the fixed ``conveyor_velocity`` magnitude.  If
    ``randomize_direction`` is True, each env independently gets +Y or -Y.

    The chosen per-env velocity is stored in ``env._belt_velocity``
    (shape: ``num_envs × 3``) so ``drive_conveyor_belt`` can reuse it.
    """
    belt: RigidObject = env.scene[belt_cfg.name]
    n = len(env_ids)

    if not hasattr(env, "_belt_velocity"):
        env._belt_velocity = torch.zeros(env.num_envs, 3, device=env.device)

    if speed_range is not None:
        lo, hi = speed_range
        speeds = torch.empty(n, device=env.device).uniform_(lo, hi)
    else:
        speeds = torch.full((n,), abs(conveyor_velocity[1]), device=env.device)

    if randomize_direction:
        # independently +1 or -1 per env
        signs = torch.sign(torch.rand(n, device=env.device) - 0.5)
    else:
        signs = torch.full((n,), -1.0 if conveyor_velocity[1] < 0 else 1.0, device=env.device)

    lin_vel = torch.zeros(n, 3, device=env.device)
    lin_vel[:, 1] = signs * speeds  # Y-axis
    env._belt_velocity[env_ids] = lin_vel

    state = wp.to_torch(belt.data.default_root_state)[env_ids].clone()
    state[:, :3] += env.scene.env_origins[env_ids]
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
    speed_range: tuple[float, float] | None = None,
    randomize_direction: bool = False,
) -> None:
    """Maintain the belt's per-env velocity and reset its pose (interval).

    Uses ``env._belt_velocity`` set by ``reset_conveyor_belt`` so each env
    keeps its own randomised speed/direction throughout the episode.
    """
    belt: RigidObject = env.scene[belt_cfg.name]
    n = len(env_ids)

    # --- reset pose to prevent translational drift ---
    state = wp.to_torch(belt.data.default_root_state)[env_ids].clone()
    state[:, :3] += env.scene.env_origins[env_ids]
    belt.write_root_pose_to_sim(state[:, :7], env_ids)

    # --- re-apply per-env velocity ---
    vel = torch.zeros(n, 6, device=env.device)
    if hasattr(env, "_belt_velocity"):
        vel[:, :3] = env._belt_velocity[env_ids]
    else:
        vel[:, :3] = _conveyor_velocity_tensor(
            conveyor_velocity, velocity_noise, n, env.device
        )
    belt.write_root_velocity_to_sim(vel, env_ids=env_ids)


# ---------------------------------------------------------------------------
# Object reset
# ---------------------------------------------------------------------------


def randomize_target_object(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    num_objects: int = 3,
) -> None:
    """Randomly assign a target object index for each resetting env.

    Stores ``env._target_object_idx`` (shape: ``num_envs``, dtype long) where
    value 0/1/2 corresponds to object_0/object_1/object_2.
    """
    if not hasattr(env, "_target_object_idx"):
        env._target_object_idx = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    env._target_object_idx[env_ids] = torch.randint(0, num_objects, (len(env_ids),), device=env.device)


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
    # Use the per-env belt velocity set by reset_conveyor_belt so objects are
    # placed moving with the (randomised speed/direction) belt and don't get a
    # friction impulse at reset. Fall back to the nominal velocity if the belt
    # hasn't been reset yet.
    if hasattr(env, "_belt_velocity"):
        belt_vel = env._belt_velocity[env_ids].clone()
        if velocity_noise > 0.0:
            mask = belt_vel.abs() > 1e-6
            perturbation = torch.empty(n, 3, device=env.device).uniform_(
                -velocity_noise, velocity_noise
            )
            belt_vel += perturbation * mask
        root_states[:, 7:10] = belt_vel
    else:
        root_states[:, 7:10] = _conveyor_velocity_tensor(
            conveyor_velocity, velocity_noise, n, env.device
        )
    root_states[:, 10:13] = 0.0

    asset.write_root_pose_to_sim(root_states[:, :7], env_ids)
    asset.write_root_velocity_to_sim(root_states[:, 7:], env_ids)


def shuffle_object_spawn_lanes(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    object_names: list[str],
) -> None:
    """Swap the (x, y) spawn spots among the objects with a per-env permutation.

    Runs *after* the per-object resets.  Each object keeps its own height (z),
    orientation, and velocity; only the horizontal landing spot is reassigned,
    so object identity is decoupled from start position.  This prevents the
    policy from learning "the target is whatever is in the front lane".
    """
    n = len(env_ids)
    num = len(object_names)
    assets = [env.scene[name] for name in object_names]

    # current world xy of each object for the resetting envs -> (n, num, 2)
    xy = torch.stack(
        [wp.to_torch(a.data.root_pos_w)[env_ids, :2] for a in assets], dim=1
    ).clone()

    # independent random permutation of lanes per env -> (n, num)
    perm = torch.argsort(torch.rand(n, num, device=env.device), dim=1)
    shuffled_xy = torch.gather(xy, 1, perm.unsqueeze(-1).expand(-1, -1, 2))

    for i, asset in enumerate(assets):
        pos = wp.to_torch(asset.data.root_pos_w)[env_ids].clone()
        quat = wp.to_torch(asset.data.root_quat_w)[env_ids].clone()
        pos[:, :2] = shuffled_xy[:, i, :]
        pose = torch.cat([pos, quat], dim=-1)
        asset.write_root_pose_to_sim(pose, env_ids)
