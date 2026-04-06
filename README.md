# Conveyor Belt Pick — Isaac Lab RL Environment

A Franka Panda learns to pick cubes from a moving conveyor belt using PPO.

Built on Isaac Lab's manager-based RL workflow. The conveyor uses friction-driven physics (same mechanism as Isaac Sim's Conveyor Belt Utility) implemented as config/events for parallel training.

## Setup

Requires Isaac Lab installed from source at `~/IsaacLab`.

```bash
# Install extra dependencies
~/IsaacLab/isaaclab.sh -p -m pip install -r ~/conveyor_belt/requirements.txt

# Install this package (editable)
~/IsaacLab/isaaclab.sh -p -m pip install -e ~/conveyor_belt

# Verify
~/IsaacLab/isaaclab.sh -p -c "import conveyor_belt; print('OK')"
```

## Usage

```bash
# Smoke test (headless)
~/IsaacLab/isaaclab.sh -p ~/conveyor_belt/conveyor_belt/test_env.py --num_envs 4

# View in browser (viser)
~/IsaacLab/isaaclab.sh -p ~/conveyor_belt/conveyor_belt/test_env.py --num_envs 1 --steps 50000 --viz viser

# Train
~/IsaacLab/isaaclab.sh -p ~/conveyor_belt/conveyor_belt/train.py --num_envs 4096 --max_iterations 5000

# Monitor
tensorboard --logdir logs/conveyor_belt
```

## File Structure

```
conveyor_belt/
├── __init__.py              # Gym registration
├── conveyor_env_cfg.py      # Top-level env config (actions, obs, rewards, events)
├── scene_cfg.py             # Scene (robot, belt, objects, rails, legs)
├── mdp/
│   ├── observations.py      # Object positions, velocities, EE vectors
│   ├── rewards.py           # Approach, lift, grasp-and-lift
│   ├── terminations.py      # Out-of-bounds, success
│   └── events.py            # Belt driving, object reset
├── agents/
│   └── rsl_rl_ppo_cfg.py   # PPO hyperparameters
├── train.py                 # Training script
└── test_env.py              # Smoke test
```
