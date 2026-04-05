# Conveyor-Belt Manipulation – Isaac Lab RL Environment

A **manager-based RL environment** for Isaac Lab where a Franka Panda robot
must pick rigid cubes from a simulated moving conveyor belt.

## Conveyor Physics Design

This environment replicates the mechanism used by the **Isaac Sim Conveyor
Belt Utility** extension (`isaacsim.asset.gen.conveyor`).  That extension
drives a rigid-body prim via an OmniGraph `IsaacConveyor` node each tick,
and objects are transported by **physics friction contact** — not by
teleporting their velocities.

We reproduce the same approach inside the Isaac Lab manager-based workflow:

| Isaac Sim Extension | This Environment |
|---|---|
| OmniGraph `IsaacConveyor` node | `drive_conveyor_belt` interval event |
| Rigid-body prim with velocity | `conveyor_belt: RigidObjectCfg` (10 000 kg, gravity off) |
| Friction-driven object transport | Belt friction 1.0 / 0.8 → objects dragged by contact |
| Per-tick velocity application | `EventTerm(mode="interval", interval_range_s=(0.05, 0.05))` |

### Why not use the OmniGraph extension directly?

Isaac Lab's manager-based RL workflow operates at a higher abstraction level
than OmniGraph.  Using `RigidObjectCfg` + an interval event keeps the
conveyor fully within the config/manager system, vectorises across all
environment instances automatically, and avoids OmniGraph overhead during
large-scale parallel training (4096+ envs).

### Upgrading to visual conveyor USD assets

To replace the placeholder cuboid with a real Isaac Sim conveyor mesh, swap
the `spawn` field in `scene_cfg.py`:

```python
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

# In ConveyorSceneCfg, replace conveyor_belt.spawn with:
spawn = UsdFileCfg(
    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Conveyors/ConveyorBelt_A09/"
             "ConveyorBelt_A09.usd",
    rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
    mass_props=sim_utils.MassPropertiesCfg(mass=10_000.0),
    collision_props=sim_utils.CollisionPropertiesCfg(),
)
```

See `Isaac/Props/Conveyors` for the full set of belt/roller/dual styles.

## Prerequisites

| Component | Version | Notes |
|-----------|---------|-------|
| NVIDIA GPU | RTX 30xx+ | Your RTX 5090 is fully supported |
| NVIDIA GPU driver | 535+ | You have 580.x — good |
| Isaac Lab (source) | 2.3.2 | Your checkout at `~/IsaacLab` |
| RSL-RL | 3.0.1+ | Installed below via pip |
| Python | 3.10+ | Via Isaac Lab's conda/venv |

Isaac Sim and all Omniverse dependencies are bundled with your Isaac Lab
source install — no separate Omniverse Launcher install is needed.

## Your Setup

```
~/
├── IsaacLab/          ← Isaac Lab 2.3.2 source install
└── conveyor_belt/     ← this project
```

## Installation

Since you have Isaac Lab installed from source at `~/IsaacLab`, all
Isaac Sim, Isaac Lab, and Omniverse packages are already available.
You only need to install the extra RL dependencies.

**Step 1 — Install additional dependencies into Isaac Lab's Python:**

```bash
~/IsaacLab/isaaclab.sh -p -m pip install rsl-rl-lib tensorboard
```

Or from the requirements file:

```bash
~/IsaacLab/isaaclab.sh -p -m pip install -r ~/conveyor_belt/requirements.txt
```

**Step 2 — Install the conveyor_belt package (editable):**

```bash
~/IsaacLab/isaaclab.sh -p -m pip install -e ~/conveyor_belt
```

This makes `conveyor_belt` importable from any directory when using
Isaac Lab's Python.

**Step 3 — Verify everything imports:**

```bash
~/IsaacLab/isaaclab.sh -p -c "import isaaclab; import rsl_rl; import conveyor_belt; print('OK')"
```

## File Structure

```
conveyor_belt/                     ← project root
├── pyproject.toml                 # Editable-install metadata
├── requirements.txt               # Extra Python dependencies (not Isaac Lab)
├── README.md
└── conveyor_belt/                 ← Python package
    ├── __init__.py                # Gym registration
    ├── conveyor_env_cfg.py        # ManagerBasedRLEnvCfg (top-level config)
    ├── scene_cfg.py               # InteractiveSceneCfg (robot, belt, objects)
    ├── mdp/
    │   ├── __init__.py            # Re-exports isaaclab.envs.mdp + custom terms
    │   ├── observations.py        # Object positions, velocities, EE vectors
    │   ├── rewards.py             # Approach, lift, grasp-and-lift rewards
    │   ├── terminations.py        # Out-of-bounds, success terminations
    │   └── events.py              # Belt driving, object reset randomisation
    ├── agents/
    │   ├── __init__.py
    │   └── rsl_rl_ppo_cfg.py     # PPO hyper-parameters (RSL-RL)
    ├── test_env.py                # Quick smoke-test (random actions)
    └── train.py                   # Full PPO training script
```

## Quick Start

**Important:** All Python commands must use Isaac Lab's Python via
`~/IsaacLab/isaaclab.sh -p` (not your system `python`).

### 1. Smoke-test the environment

Run `test_env.py` first to verify the scene loads and physics works:

```bash
# Headless (fastest — no GUI, good for SSH / CI)
~/IsaacLab/isaaclab.sh -p ~/conveyor_belt/conveyor_belt/test_env.py --num_envs 4 --headless

# With the Omniverse viewport (see the robot + belt + cubes)
~/IsaacLab/isaaclab.sh -p ~/conveyor_belt/conveyor_belt/test_env.py --num_envs 4

# Longer test
~/IsaacLab/isaaclab.sh -p ~/conveyor_belt/conveyor_belt/test_env.py --num_envs 16 --steps 500
```

Expected output:

```
 Creating env with 4 instances ...

  Observation space : Box(...)
  Action space      : Box(...)
  Obs tensor shape  : torch.Size([4, 53])

  Running 200 steps with random actions ...

    step   50  |  mean cumulative reward:    3.142  |  episodes done so far: 0
    step  100  |  mean cumulative reward:    5.871  |  episodes done so far: 2
    ...

  --- Physics snapshot (env 0) ---
  Belt position : [0.5, 0.0, 0.0]
  Belt velocity : [0.0, -0.08, 0.0]
  ...
```

**What to look for:**
- Obs shape should be `[num_envs, 53]` (9+9+9+9+9+8).
- Rewards should be positive (approach reward fires even with random actions).
- Belt velocity should be close to `[0, -0.08, 0]`.
- Belt position should stay near `[0.5, 0, 0]` (no drift).
- Object velocities should be near the belt velocity while on the belt.
- Episodes should complete (via timeout at 8 s or out-of-bounds).

### 2. Train with PPO

```bash
# Standard training (256 envs)
~/IsaacLab/isaaclab.sh -p ~/conveyor_belt/conveyor_belt/train.py --num_envs 256 --max_iterations 1500

# Full-scale training on your RTX 5090
~/IsaacLab/isaaclab.sh -p ~/conveyor_belt/conveyor_belt/train.py --num_envs 4096 --max_iterations 3000

# Record evaluation videos
~/IsaacLab/isaaclab.sh -p ~/conveyor_belt/conveyor_belt/train.py --num_envs 64 --video
```

Logs (TensorBoard events, checkpoints) are written to
`./logs/conveyor_belt/<timestamp>/`.

### 3. Monitor training with TensorBoard

```bash
tensorboard --logdir logs/conveyor_belt
```

Then open `http://localhost:6006`. Key metrics:

* **Episode reward** – mean/total return per episode.
* **Policy / Value loss** – PPO loss curves.
* **Success rate** – driven by the `success` termination term.

## Architecture Overview

### Scene (`scene_cfg.py`)

* **Franka Panda** – 7-DOF arm + parallel gripper via `FRANKA_PANDA_CFG`.
* **FrameTransformer** – tracks end-effector world position.
* **Conveyor belt** – a tracked `RigidObjectCfg` (10 000 kg, gravity
  disabled, high friction).  Its velocity is maintained by an interval
  event, mirroring the Isaac Sim Conveyor Belt Utility.
* **3 rigid cubes** – spawned with distinct colours (red / green / blue).
  They ride on the belt via friction contact.
* Ground plane + dome light.

### Conveyor Simulation

1. At **reset**, the belt is restored to its default pose and given a
   randomised velocity (~0.08 m/s along −y ± noise).  Each object is also
   spawned with a matching initial velocity to avoid contact impulses.
2. An **interval event** (`drive_conveyor_belt`) fires every 0.05 s to
   re-apply the target velocity to the belt rigid body – the Isaac Lab
   equivalent of the `IsaacConveyor` OmniGraph tick.
3. The belt has **high friction** (static 1.0, dynamic 0.8) so objects
   resting on it are dragged by physics contact.  Grasped/lifted objects
   naturally separate from the belt since they are no longer in contact.
4. The belt's enormous mass (10 000 kg) with gravity disabled means
   object contacts cause negligible deceleration or drift.

### Observations

| Term | Dim | Source |
|------|-----|--------|
| Joint positions (relative) | 9 | `mdp.joint_pos_rel` |
| Joint velocities (relative) | 9 | `mdp.joint_vel_rel` |
| Object positions (robot frame) | 9 | custom |
| Object linear velocities | 9 | custom |
| EE → object vectors | 9 | custom |
| Last action | 8 | `mdp.last_action` |

### Actions

* **Arm** – 7-DOF joint-position offsets (scaled 0.5×).
* **Gripper** – binary open (0.04 m) / close (0.0 m).
* Total action dim: **8**.

### Rewards

| Term | Weight | Description |
|------|--------|-------------|
| `approach_object` | 1.0 | tanh kernel on closest EE ↔ object distance |
| `lift_object` | 15.0 | any object above 0.08 m |
| `grasp_lift` | 10.0 | object near EE *and* above 0.08 m |
| `action_rate` | −1e-4 | L2 action-rate penalty |
| `joint_vel` | −1e-4 | L2 joint-velocity penalty |

### Terminations

| Term | Type |
|------|------|
| `time_out` | timeout (8 s episode) |
| `object_out_of_bounds` | real done – any object leaves workspace |
| `success` | real done – object lifted to 0.20 m near EE |

### Events

| Event | Mode | Purpose |
|-------|------|---------|
| `reset_scene` | reset | restore defaults |
| `reset_belt` | reset | reset belt pose + randomise velocity |
| `reset_object_*` | reset | randomise each object's pose / yaw / velocity |
| `conveyor_drive` | interval (0.05 s) | maintain belt velocity (≈ OmniGraph tick) |

## Next Steps for Improving Training

1. **Visual conveyor assets** – swap in USD meshes from
   `Isaac/Props/Conveyors` for photorealistic rendering and conveyor
   texture animation.
2. **OmniGraph integration** – for non-RL workflows (e.g. data
   collection), use the full `isaacsim.asset.gen.conveyor` extension
   with its `IsaacConveyor` OmniGraph node for native texture animation
   and UI controls.
3. **Curriculum on conveyor speed** – start slow, increase as policy
   improves (`CurriculumTermCfg` + `modify_reward_weight`).
4. **Domain randomisation** – vary object sizes, masses, friction, and
   belt speed via `startup`/`reset` events.
5. **IK actions** – switch to `DifferentialInverseKinematicsActionCfg`
   for easier Cartesian exploration.
6. **Asymmetric critic** – give the value function privileged info
   (e.g. contact forces via `body_incoming_wrench`).
7. **Multi-object curriculum** – train with 1 object first, add more
   once the base grasp is learned.
8. **Longer training** – 3000-5000 iterations with 4096 envs; tune
   `entropy_coef` and LR schedule.
9. **VLA data collection** – once success rate exceeds ~60 %, run the
   trained checkpoint with cameras enabled to collect image-action
   trajectories for downstream vision-language-action models.
