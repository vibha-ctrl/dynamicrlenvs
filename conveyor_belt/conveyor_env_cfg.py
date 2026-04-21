"""Manager-based RL environment configuration for conveyor-belt manipulation.

A Franka Panda must pick YCB objects from a simulated moving conveyor belt.

Conveyor physics
~~~~~~~~~~~~~~~~
The belt is a tracked ``RigidObjectCfg`` whose velocity is maintained by an
interval event (``drive_conveyor_belt``).  This mirrors the mechanism used
by the Isaac Sim *Conveyor Belt Utility* extension
(``isaacsim.asset.gen.conveyor``), which drives a rigid-body prim via an
OmniGraph ``IsaacConveyor`` node each tick.  Objects are transported by
**physics friction contact** with the belt, not by having their velocities
overwritten.
"""

from __future__ import annotations

from isaaclab_physx.physics import PhysxCfg

from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from . import mdp
from .scene_cfg import ConveyorSceneCfg

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

OBJECT_NAMES: list[str] = ["object_0", "object_1", "object_2"]

CONVEYOR_SPEED_RANGE = (0.03, 0.10)  # m/s — per-env speed sampled uniformly
CONVEYOR_BELT_NOISE = 0.005         # ± m/s — belt speed variation (small)
CONVEYOR_OBJECT_NOISE = 0.02        # ± m/s — per-object spawn velocity spread


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------


@configclass
class ActionsCfg:
    """Joint-position control for the arm and binary open/close for the gripper."""

    arm_action: mdp.JointPositionActionCfg = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        scale=0.5,
        use_default_offset=True,
    )
    gripper_action: mdp.BinaryJointPositionActionCfg = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_finger.*"],
        open_command_expr={"panda_finger_.*": 0.04},
        close_command_expr={"panda_finger_.*": 0.0},
    )


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------


@configclass
class ObservationsCfg:
    """Observation specification."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations fed to the policy network."""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        object_positions = ObsTerm(
            func=mdp.object_positions_in_robot_frame,
            params={"object_names": OBJECT_NAMES},
        )
        object_velocities = ObsTerm(
            func=mdp.object_linear_velocities,
            params={"object_names": OBJECT_NAMES},
        )
        ee_to_objects = ObsTerm(
            func=mdp.ee_to_object_vectors,
            params={"object_names": OBJECT_NAMES},
        )
        conveyor_velocity = ObsTerm(func=mdp.conveyor_velocity)
        target_object = ObsTerm(
            func=mdp.target_object_one_hot,
            params={"num_objects": len(OBJECT_NAMES)},
        )
        last_actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


# ---------------------------------------------------------------------------
# Events (randomisation & conveyor physics)
# ---------------------------------------------------------------------------


_BELT_RESET_PARAMS = {
    "belt_cfg": SceneEntityCfg("conveyor_belt"),
    "speed_range": CONVEYOR_SPEED_RANGE,
    "velocity_noise": CONVEYOR_BELT_NOISE,
}

_BELT_DRIVE_PARAMS = {
    "belt_cfg": SceneEntityCfg("conveyor_belt"),
}


_UPRIGHT_QUATS = {
    "object_0": (-0.7071, 0.0, 0.0, 0.7071),  # -90° about X in xyzw
    "object_1": (-0.7071, 0.0, 0.0, 0.7071),  # -90° about X in xyzw
    "object_2": (-0.7071, 0.0, 0.0, 0.7071),  # -90° about X in xyzw
}


def _object_reset_term(asset_name: str) -> EventTerm:
    """Factory for per-object reset events."""
    return EventTerm(
        func=mdp.reset_object_on_conveyor,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(asset_name),
            "pose_range": {"x": (-0.08, 0.08), "y": (-0.15, 0.15)},
            "yaw_range": (0.0, 0.0),
            "velocity_noise": CONVEYOR_OBJECT_NOISE,
            "upright_quat": _UPRIGHT_QUATS[asset_name],
        },
    )


@configclass
class EventCfg:
    """Randomisation events and conveyor-belt driving."""

    # -- scene / belt reset ---------------------------------------------------
    reset_scene = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_belt = EventTerm(
        func=mdp.reset_conveyor_belt,
        mode="reset",
        params=_BELT_RESET_PARAMS,
    )

    # -- target object selection -----------------------------------------------
    reset_target = EventTerm(
        func=mdp.randomise_target_object,
        mode="reset",
        params={"num_objects": len(OBJECT_NAMES)},
    )

    # -- per-object reset (position / orientation / velocity) -----------------
    reset_object_0 = _object_reset_term("object_0")
    reset_object_1 = _object_reset_term("object_1")
    reset_object_2 = _object_reset_term("object_2")

    # -- conveyor drive (interval, ~20 Hz) ------------------------------------
    # Mirrors the per-tick OmniGraph evaluation of the IsaacConveyor node
    # in the Isaac Sim Conveyor Belt Utility.
    conveyor_drive = EventTerm(
        func=mdp.drive_conveyor_belt,
        mode="interval",
        interval_range_s=(0.05, 0.05),
        params=_BELT_DRIVE_PARAMS,
    )


# ---------------------------------------------------------------------------
# Rewards
# ---------------------------------------------------------------------------


@configclass
class RewardsCfg:
    """Progress-based rewards: pay for *improving*, not for *being*.

    - approach: dense, turns off after grasp
    - grasp_event: one-time bonus on first grasp
    - lift_progress: delta-height reward (zero for hovering)
    - milestones: one-time bonuses at 8/12/16 cm
    - success_reward: large terminal bonus (gated on controlled grasp)
    - time_cost: small per-step penalty to encourage finishing fast
    """

    approach_object = RewTerm(
        func=mdp.approach_object,
        params={"std": 0.1, "object_names": OBJECT_NAMES},
        weight=2.0,
    )

    grasp_event = RewTerm(
        func=mdp.grasp_event,
        params={
            "minimal_height": 0.08,
            "max_grasp_distance": 0.08,
            "object_names": OBJECT_NAMES,
        },
        weight=5.0,
    )

    lift_progress = RewTerm(
        func=mdp.lift_progress,
        params={"object_names": OBJECT_NAMES},
        weight=100.0,
    )

    success_reward = RewTerm(
        func=mdp.success_bonus,
        params={
            "target_height": 0.20,
            "max_grasp_distance": 0.12,
            "object_names": OBJECT_NAMES,
        },
        weight=500.0,
    )

    time_cost = RewTerm(func=mdp.alive_cost, weight=-0.01)

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


# ---------------------------------------------------------------------------
# Terminations
# ---------------------------------------------------------------------------


@configclass
class TerminationsCfg:
    """Episode termination conditions."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_out_of_bounds = DoneTerm(
        func=mdp.all_objects_out_of_bounds,
        params={
            "object_names": OBJECT_NAMES,
            "x_bounds": (-0.1, 1.0),
            "y_bounds": (-1.2, 1.2),
            "z_min": -0.1,
        },
    )

    success = DoneTerm(
        func=mdp.successful_grasp_lift,
        params={
            "target_height": 0.20,
            "max_grasp_distance": 0.12,
            "object_names": OBJECT_NAMES,
        },
    )


# ---------------------------------------------------------------------------
# Top-level environment config
# ---------------------------------------------------------------------------


@configclass
class ConveyorBeltEnvCfg(ManagerBasedRLEnvCfg):
    """Full configuration for the conveyor-belt pick environment."""

    scene: ConveyorSceneCfg = ConveyorSceneCfg(num_envs=4096, env_spacing=2.5)
    viewer: ViewerCfg = ViewerCfg(
        eye=(1.5, -1.5, 1.0),
        lookat=(0.5, 0.0, 0.2),
        resolution=(1280, 720),
    )
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self) -> None:
        """Post-initialisation: simulation parameters."""
        self.decimation = 2
        self.episode_length_s = 20.0
        # sim
        self.sim.dt = 0.01  # 100 Hz physics
        self.sim.render_interval = self.decimation

        self.sim.physics = PhysxCfg(
            bounce_threshold_velocity=0.2,
            gpu_found_lost_aggregate_pairs_capacity=1024 * 1024 * 4,
            gpu_total_aggregate_pairs_capacity=16 * 1024,
            friction_correlation_distance=0.00625,
        )
