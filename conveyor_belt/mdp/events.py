"""Custom event terms for the conveyor-belt manipulation task.

Conveyor physics
~~~~~~~~~~~~~~~~
The belt is a **high-mass dynamic rigid body** (gravity disabled) whose
velocity is maintained by an interval event via the tensor API.  High mass
makes it effectively immovable by friction while still having a real PhysX
simulated velocity that the contact solver uses to drag objects.

* ``reset_conveyor_belt``  – sets belt pose + velocity on episode reset.
* ``drive_conveyor_belt``  – interval event (~20 Hz) that resets pose and
  re-applies velocity each tick, preventing slow translational drift caused
  by reaction forces from objects resting on the belt.
* ``reset_object_on_conveyor`` – randomises each object's pose and gives
  it an initial velocity matching the belt so there is no contact impulse
  at spawn.

Objects are transported by **physics friction** with the belt.

Note: ``PhysxSurfaceVelocityAPI`` was tested and rejected — it breaks
collision entirely in Isaac Lab headless/Gym mode (upstream issue #4561,
present in Isaac Sim 5.1 and 6.0).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject
    from isaaclab.envs import ManagerBasedRLEnv


def _quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product of two quaternion batches ``(N, 4)`` in ``[w, x, y, z]`` order."""
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=-1,
    )


def _get_or_init_conveyor_vel(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return the ``(num_envs, 3)`` per-env conveyor velocity buffer.

    Lazily allocated on first access so the buffer always exists when any
    event or observation term reads it.
    """
    if not hasattr(env, "_conveyor_vel_per_env"):
        env._conveyor_vel_per_env = torch.zeros(
            env.num_envs, 3, device=env.device
        )
    return env._conveyor_vel_per_env


def _randomise_belt_velocity(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    speed_range: tuple[float, float],
    velocity_noise: float,
) -> torch.Tensor:
    """Sample per-env belt velocity: random speed in ``speed_range``, random +Y/-Y direction.

    The result is written into ``env._conveyor_vel_per_env`` for the given
    ``env_ids`` and also returned.
    """
    buf = _get_or_init_conveyor_vel(env)
    n = len(env_ids)

    speed = torch.empty(n, device=env.device).uniform_(*speed_range)
    direction = torch.where(
        torch.randint(0, 2, (n,), device=env.device).bool(),
        torch.ones(n, device=env.device),
        -torch.ones(n, device=env.device),
    )
    vel_y = speed * direction

    if velocity_noise > 0.0:
        vel_y += torch.empty(n, device=env.device).uniform_(-velocity_noise, velocity_noise)

    vel = torch.zeros(n, 3, device=env.device)
    vel[:, 1] = vel_y
    buf[env_ids] = vel
    return vel


# ---------------------------------------------------------------------------
# Belt events (PhysxSurfaceVelocityAPI approach)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Belt events
# ---------------------------------------------------------------------------


def reset_conveyor_belt(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    belt_cfg: SceneEntityCfg,
    speed_range: tuple[float, float],
    velocity_noise: float,
) -> None:
    """Reset the conveyor belt to its default pose with per-env randomised velocity."""
    belt: RigidObject = env.scene[belt_cfg.name]

    state = wp.to_torch(belt.data.default_root_state)[env_ids].clone()
    state[:, :3] += env.scene.env_origins[env_ids]

    lin_vel = _randomise_belt_velocity(env, env_ids, speed_range, velocity_noise)
    state[:, 7:10] = lin_vel
    state[:, 10:13] = 0.0

    belt.write_root_pose_to_sim(state[:, :7], env_ids)
    belt.write_root_velocity_to_sim(state[:, 7:], env_ids)


def drive_conveyor_belt(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    belt_cfg: SceneEntityCfg,
) -> None:
    """Maintain belt velocity and reset pose each interval tick.

    Kinematic bodies don't drift from friction but do translate from the
    imposed velocity, so the pose reset keeps the belt in place while the
    velocity keeps driving contact friction on objects above it.
    """
    belt: RigidObject = env.scene[belt_cfg.name]
    buf = _get_or_init_conveyor_vel(env)

    state = wp.to_torch(belt.data.default_root_state)[env_ids].clone()
    state[:, :3] += env.scene.env_origins[env_ids]
    belt.write_root_pose_to_sim(state[:, :7], env_ids)

    vel = torch.zeros(len(env_ids), 6, device=env.device)
    vel[:, :3] = buf[env_ids]
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
    velocity_noise: float,
    upright_quat: tuple[float, float, float, float] = (-0.7071, 0.0, 0.0, 0.7071),
) -> None:
    """Reset one rigid object to a random pose on the conveyor belt.

    * Position = default init position + uniform offset per axis.
    * Orientation = fixed ``upright_quat`` (xyzw quaternion order).
    * Linear velocity = per-env belt velocity (from ``env._conveyor_vel_per_env``)
      ± small noise so there is no impulse when the object is placed on the belt.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    n = len(env_ids)

    root_states = wp.to_torch(asset.data.default_root_state)[env_ids].clone()

    # --- position ---
    for i, key in enumerate(("x", "y", "z")):
        lo, hi = pose_range.get(key, (0.0, 0.0))
        root_states[:, i] += torch.empty(n, device=env.device).uniform_(lo, hi)
    root_states[:, :3] += env.scene.env_origins[env_ids]

    # --- force upright orientation ---
    base_quat = torch.tensor(upright_quat, device=env.device, dtype=torch.float32)
    base_quat = base_quat.unsqueeze(0).expand(n, -1)
    root_states[:, 3:7] = base_quat

    # --- match per-env belt velocity + small noise ---
    buf = _get_or_init_conveyor_vel(env)
    obj_vel = buf[env_ids].clone()
    if velocity_noise > 0.0:
        noise_mask = obj_vel.abs() > 1e-6
        perturbation = torch.empty(n, 3, device=env.device).uniform_(-velocity_noise, velocity_noise)
        obj_vel[noise_mask] += perturbation[noise_mask]
    root_states[:, 7:10] = obj_vel
    root_states[:, 10:13] = 0.0

    asset.write_root_pose_to_sim(root_states[:, :7], env_ids)
    asset.write_root_velocity_to_sim(root_states[:, 7:], env_ids)


# ---------------------------------------------------------------------------
# Target-object randomisation
# ---------------------------------------------------------------------------


def get_target_object_idx(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return the ``(num_envs,)`` buffer of per-env target object indices.

    Lazily initialised to zeros; updated by ``randomise_target_object``.
    """
    if not hasattr(env, "_target_object_idx"):
        env._target_object_idx = torch.zeros(
            env.num_envs, device=env.device, dtype=torch.long
        )
    return env._target_object_idx


def randomise_target_object(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    num_objects: int = 3,
) -> None:
    """Randomly choose which object each env must pick this episode."""
    buf = get_target_object_idx(env)
    buf[env_ids] = torch.randint(0, num_objects, (len(env_ids),), device=env.device)
