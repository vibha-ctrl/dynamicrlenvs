"""Scene configuration for the conveyor-belt manipulation environment.

The conveyor belt is modelled as a **high-mass dynamic rigid body** whose
velocity is driven externally (by an interval event).  This mirrors the
mechanism used by the Isaac Sim *Conveyor Belt Utility* extension
(``isaacsim.asset.gen.conveyor``), which applies velocity to a rigid-body
prim via an OmniGraph node each tick.  Objects resting on the belt are
transported by **physics friction contact**, not by teleporting their
velocities directly.

To swap in a visual conveyor USD instead of the placeholder cuboid, replace
the ``spawn`` field of ``conveyor_belt`` with a ``UsdFileCfg`` pointing at
one of the assets shipped with Isaac Sim, e.g.::

    from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
    from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

    spawn = UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Conveyors/ConveyorBelt_A09/"
                 "ConveyorBelt_A09.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
        ),
    )

and adjust the ``init_state`` / ``size`` to match the asset geometry.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG  # isort: skip

# ---------------------------------------------------------------------------
# Belt geometry
# ---------------------------------------------------------------------------

BELT_WIDTH = 0.4
BELT_LENGTH = 2.0
BELT_THICKNESS = 0.02
BELT_CENTER_X = 0.5

CONVEYOR_SURFACE_Z = 0.0

_BELT_TOP_Z = CONVEYOR_SURFACE_Z + BELT_THICKNESS / 2
_BELT_BOTTOM_Z = CONVEYOR_SURFACE_Z - BELT_THICKNESS / 2
_BELT_LEFT_X = BELT_CENTER_X - BELT_WIDTH / 2
_BELT_RIGHT_X = BELT_CENTER_X + BELT_WIDTH / 2

# ---------------------------------------------------------------------------
# Side rails
# ---------------------------------------------------------------------------

RAIL_HEIGHT = 0.04
RAIL_THICKNESS = 0.015
_RAIL_Z = _BELT_TOP_Z + RAIL_HEIGHT / 2
_LEFT_RAIL_X = _BELT_LEFT_X - RAIL_THICKNESS / 2
_RIGHT_RAIL_X = _BELT_RIGHT_X + RAIL_THICKNESS / 2

# ---------------------------------------------------------------------------
# Support legs
# ---------------------------------------------------------------------------

GROUND_Z = -1.05
SUPPORT_HEIGHT = _BELT_BOTTOM_Z - GROUND_Z  # legs reach from belt bottom to ground
SUPPORT_THICKNESS = 0.03
_LEG_Z = _BELT_BOTTOM_Z - SUPPORT_HEIGHT / 2
_LEG_INSET_X = 0.04
_LEG_INSET_Y = 0.06

# ---------------------------------------------------------------------------
# Robot pedestal
# ---------------------------------------------------------------------------

ROBOT_BASE_Z = -0.12
_PEDESTAL_WIDTH = 0.28
_PEDESTAL_DEPTH = 0.28
_PEDESTAL_HEIGHT = ROBOT_BASE_Z - GROUND_Z  # from ground up to robot base
_PEDESTAL_Z = GROUND_Z + _PEDESTAL_HEIGHT / 2
_PEDESTAL_COLOR = (0.35, 0.35, 0.38)

# ---------------------------------------------------------------------------
# Objects (YCB items from Nucleus)
# ---------------------------------------------------------------------------

_YCB_ROOT = f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics"

_COMMON_RIGID_PROPS = sim_utils.RigidBodyPropertiesCfg(
    solver_position_iteration_count=16,
    solver_velocity_iteration_count=1,
    max_angular_velocity=100.0,
    max_linear_velocity=10.0,
    max_depenetration_velocity=1.0,
    disable_gravity=False,
)

_BELT_TOP = CONVEYOR_SURFACE_Z + BELT_THICKNESS / 2

OBJECT_SCALE = (0.65, 0.65, 0.65)

OBJECT_HALF_HEIGHTS: dict[str, float] = {
    "sugar_box": 0.088 * OBJECT_SCALE[2],
    "soup_can": 0.051 * OBJECT_SCALE[2],
    "mustard_bottle": 0.096 * OBJECT_SCALE[2],
}

_UPRIGHT_QUATS = {
    "sugar_box":      [-0.7071, 0.0, 0.0, 0.7071],  # -90° about X in xyzw
    "soup_can":       [-0.7071, 0.0, 0.0, 0.7071],  # -90° about X in xyzw
    "mustard_bottle": [-0.7071, 0.0, 0.0, 0.7071],  # -90° about X in xyzw
}

OBJECT_SPAWN_Z_SUGAR = _BELT_TOP + OBJECT_HALF_HEIGHTS["sugar_box"] + 0.002
OBJECT_SPAWN_Z_SOUP = _BELT_TOP + OBJECT_HALF_HEIGHTS["soup_can"] + 0.002
OBJECT_SPAWN_Z_MUSTARD = _BELT_TOP + OBJECT_HALF_HEIGHTS["mustard_bottle"] + 0.002


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ee_marker_cfg = FRAME_MARKER_CFG.copy()
_ee_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
_ee_marker_cfg.prim_path = "/Visuals/FrameTransformer"


def _ycb_spawn_cfg(usd_name: str, mass: float = 0.2) -> UsdFileCfg:
    """Return a UsdFileCfg for a YCB object on the conveyor."""
    return UsdFileCfg(
        usd_path=f"{_YCB_ROOT}/{usd_name}",
        scale=OBJECT_SCALE,
        rigid_props=_COMMON_RIGID_PROPS,
        mass_props=sim_utils.MassPropertiesCfg(mass=mass),
        collision_props=sim_utils.CollisionPropertiesCfg(),
    )


def _kinematic_cuboid(
    size: tuple[float, float, float],
    color: tuple[float, float, float],
    *,
    collision: bool = False,
) -> sim_utils.CuboidCfg:
    """Return a kinematic CuboidCfg for structural conveyor parts."""
    kwargs = dict(
        size=size,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=True,
            disable_gravity=True,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
    )
    if collision:
        kwargs["collision_props"] = sim_utils.CollisionPropertiesCfg()
        kwargs["physics_material"] = sim_utils.RigidBodyMaterialCfg(
            static_friction=0.3,
            dynamic_friction=0.2,
        )
    return sim_utils.CuboidCfg(**kwargs)


_RAIL_COLOR = (0.18, 0.18, 0.20)
_LEG_COLOR = (0.30, 0.30, 0.33)
_BELT_COLOR = (0.22, 0.22, 0.26)


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------


@configclass
class ConveyorSceneCfg(InteractiveSceneCfg):
    """Scene with a Franka Panda, a driven conveyor belt, and three YCB objects.

    The belt is a **tracked RigidObject** so its velocity can be set by an
    event term – the same approach the Isaac Sim Conveyor Belt Utility uses
    internally (``isaacsim.asset.gen.conveyor``).
    """

    # -- Robot ----------------------------------------------------------------
    robot: ArticulationCfg = FRANKA_PANDA_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=[0.05, 0.0, ROBOT_BASE_Z],
            joint_pos={
                "panda_joint1": 0.0,
                "panda_joint2": -0.569,
                "panda_joint3": 0.0,
                "panda_joint4": -2.810,
                "panda_joint5": 0.0,
                "panda_joint6": 3.037,
                "panda_joint7": 0.741,
                "panda_finger_joint.*": 0.04,
            },
        ),
    )

    # -- Robot pedestal (visual support) --------------------------------------
    robot_pedestal: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/RobotPedestal",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[0.0, 0.0, _PEDESTAL_Z],
        ),
        spawn=_kinematic_cuboid(
            size=(_PEDESTAL_WIDTH, _PEDESTAL_DEPTH, _PEDESTAL_HEIGHT),
            color=_PEDESTAL_COLOR,
        ),
    )

    # -- End-effector frame sensor --------------------------------------------
    ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
        debug_vis=False,
        visualizer_cfg=_ee_marker_cfg,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                name="end_effector",
                offset=OffsetCfg(pos=[0.0, 0.0, 0.1034]),
            ),
        ],
    )

    # -- Conveyor belt (driven rigid body) ------------------------------------
    conveyor_belt: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/ConveyorBelt",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[BELT_CENTER_X, 0.0, CONVEYOR_SURFACE_Z],
            rot=[0, 0, 0, 1],
            lin_vel=[0.0, 0.0, 0.0],  # randomised per-env at reset
        ),
        spawn=sim_utils.CuboidCfg(
            size=(BELT_WIDTH, BELT_LENGTH, BELT_THICKNESS),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_linear_velocity=0.5,
                max_angular_velocity=0.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=10_000.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=0.8,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=_BELT_COLOR),
        ),
    )

    # -- Side rails (kinematic, with collision to keep objects on belt) --------
    rail_left: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/RailLeft",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[_LEFT_RAIL_X, 0.0, _RAIL_Z],
        ),
        spawn=_kinematic_cuboid(
            size=(RAIL_THICKNESS, BELT_LENGTH, RAIL_HEIGHT),
            color=_RAIL_COLOR,
            collision=True,
        ),
    )

    rail_right: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/RailRight",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[_RIGHT_RAIL_X, 0.0, _RAIL_Z],
        ),
        spawn=_kinematic_cuboid(
            size=(RAIL_THICKNESS, BELT_LENGTH, RAIL_HEIGHT),
            color=_RAIL_COLOR,
            collision=True,
        ),
    )

    # -- Support legs (kinematic, visual only) --------------------------------
    leg_fl: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/LegFL",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[
                _BELT_LEFT_X + _LEG_INSET_X,
                -BELT_LENGTH / 2 + _LEG_INSET_Y,
                _LEG_Z,
            ],
        ),
        spawn=_kinematic_cuboid(
            size=(SUPPORT_THICKNESS, SUPPORT_THICKNESS, SUPPORT_HEIGHT),
            color=_LEG_COLOR,
        ),
    )

    leg_fr: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/LegFR",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[
                _BELT_RIGHT_X - _LEG_INSET_X,
                -BELT_LENGTH / 2 + _LEG_INSET_Y,
                _LEG_Z,
            ],
        ),
        spawn=_kinematic_cuboid(
            size=(SUPPORT_THICKNESS, SUPPORT_THICKNESS, SUPPORT_HEIGHT),
            color=_LEG_COLOR,
        ),
    )

    leg_bl: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/LegBL",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[
                _BELT_LEFT_X + _LEG_INSET_X,
                BELT_LENGTH / 2 - _LEG_INSET_Y,
                _LEG_Z,
            ],
        ),
        spawn=_kinematic_cuboid(
            size=(SUPPORT_THICKNESS, SUPPORT_THICKNESS, SUPPORT_HEIGHT),
            color=_LEG_COLOR,
        ),
    )

    leg_br: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/LegBR",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[
                _BELT_RIGHT_X - _LEG_INSET_X,
                BELT_LENGTH / 2 - _LEG_INSET_Y,
                _LEG_Z,
            ],
        ),
        spawn=_kinematic_cuboid(
            size=(SUPPORT_THICKNESS, SUPPORT_THICKNESS, SUPPORT_HEIGHT),
            color=_LEG_COLOR,
        ),
    )

    # -- Rigid objects on conveyor (YCB items) ---------------------------------
    # Objects start in the upper half of the belt (positive Y) and travel
    # toward the robot in the -Y direction.
    object_0: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object_0",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.50, 0.35, OBJECT_SPAWN_Z_SUGAR],
            rot=_UPRIGHT_QUATS["sugar_box"],
        ),
        spawn=_ycb_spawn_cfg("004_sugar_box.usd", mass=0.5),
    )

    object_1: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object_1",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.45, 0.15, OBJECT_SPAWN_Z_SOUP],
            rot=_UPRIGHT_QUATS["soup_can"],
        ),
        spawn=_ycb_spawn_cfg("005_tomato_soup_can.usd", mass=0.35),
    )

    object_2: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object_2",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.55, -0.05, OBJECT_SPAWN_Z_MUSTARD],
            rot=_UPRIGHT_QUATS["mustard_bottle"],
        ),
        spawn=_ycb_spawn_cfg("006_mustard_bottle.usd", mass=0.4),
    )

    # -- Ground plane ---------------------------------------------------------
    plane: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.0, -1.05]),
        spawn=GroundPlaneCfg(),
    )

    # -- Dome light -----------------------------------------------------------
    light: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
