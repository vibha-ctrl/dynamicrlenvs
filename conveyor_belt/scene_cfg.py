# Copyright (c) 2024-2026. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

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
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.utils import configclass

from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG  # isort: skip

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OBJECT_SIZE = (0.04, 0.04, 0.04)
OBJECT_MASS_KG = 0.1
CONVEYOR_SURFACE_Z = 0.0
CONVEYOR_BELT_THICKNESS = 0.02
OBJECT_SPAWN_Z = (
    CONVEYOR_SURFACE_Z + CONVEYOR_BELT_THICKNESS / 2 + OBJECT_SIZE[2] / 2
    + 0.002  # small clearance to avoid first-frame interpenetration with belt
)

# Default belt velocity (m/s).  -y = objects travel right-to-left.
CONVEYOR_VELOCITY = (0.0, -0.08, 0.0)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ee_marker_cfg = FRAME_MARKER_CFG.copy()
_ee_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
_ee_marker_cfg.prim_path = "/Visuals/FrameTransformer"


def _object_spawn_cfg(color: tuple[float, float, float]) -> sim_utils.CuboidCfg:
    """Return a CuboidCfg for one conveyor object."""
    return sim_utils.CuboidCfg(
        size=OBJECT_SIZE,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_angular_velocity=100.0,
            max_linear_velocity=10.0,
            max_depenetration_velocity=1.0,
            disable_gravity=False,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=OBJECT_MASS_KG),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.5,
            dynamic_friction=0.3,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
    )


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------


@configclass
class ConveyorSceneCfg(InteractiveSceneCfg):
    """Scene with a Franka Panda, a driven conveyor belt, and three rigid cubes.

    The belt is a **tracked RigidObject** so its velocity can be set by an
    event term – the same approach the Isaac Sim Conveyor Belt Utility uses
    internally (``isaacsim.asset.gen.conveyor``).
    """

    # -- Robot ----------------------------------------------------------------
    robot: ArticulationCfg = FRANKA_PANDA_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
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
    #
    # Physics design (mirrors the Isaac Sim Conveyor Belt Utility):
    #   * Dynamic rigid body so ``write_root_velocity_to_sim`` has effect.
    #   * Very high mass  ➜  objects cannot push the belt off course.
    #   * Gravity disabled ➜  belt does not sink or drift vertically.
    #   * HIGH friction    ➜  objects are dragged by contact, not teleported.
    #
    # An interval event calls ``drive_conveyor_belt`` every 0.05 s to
    # maintain the target velocity, analogous to the OmniGraph tick in the
    # Isaac Sim extension.
    conveyor_belt: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/ConveyorBelt",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.5, 0.0, CONVEYOR_SURFACE_Z],
            rot=[1, 0, 0, 0],
            lin_vel=list(CONVEYOR_VELOCITY),
        ),
        spawn=sim_utils.CuboidCfg(
            size=(0.3, 0.8, CONVEYOR_BELT_THICKNESS),
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
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.25, 0.25, 0.30),
            ),
        ),
    )

    # -- Rigid objects on conveyor --------------------------------------------
    object_0: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object_0",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.50, 0.25, OBJECT_SPAWN_Z],
            rot=[1, 0, 0, 0],
        ),
        spawn=_object_spawn_cfg(color=(0.85, 0.15, 0.15)),
    )

    object_1: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object_1",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.45, 0.10, OBJECT_SPAWN_Z],
            rot=[1, 0, 0, 0],
        ),
        spawn=_object_spawn_cfg(color=(0.15, 0.75, 0.15)),
    )

    object_2: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object_2",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.55, -0.05, OBJECT_SPAWN_Z],
            rot=[1, 0, 0, 0],
        ),
        spawn=_object_spawn_cfg(color=(0.15, 0.15, 0.85)),
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
