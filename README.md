# Conveyor Belt Pick — Isaac Lab RL Environment

A Franka Panda learns to pick YCB objects (sugar box, soup can, mustard bottle) from a moving conveyor belt using PPO.

Built on Isaac Lab's manager-based RL workflow. The conveyor uses friction-driven physics (same mechanism as Isaac Sim's Conveyor Belt Utility) implemented as config/events for parallel training.

Each parallel environment gets a randomized conveyor speed/direction and a randomly assigned target object, forcing the policy to generalize across object types and belt conditions.

## Setup

Requires Isaac Lab installed from source at `~/IsaacLab`.

```bash
# Create and activate a virtual environment
cd ~/IsaacLab
uv venv vibhaenv --python 3.12
source vibhaenv/bin/activate

# Install Isaac Sim
uv pip install 'isaacsim[all]==6.0.0' --extra-index-url https://pypi.nvidia.com

# Install extra dependencies
uv pip install -r ~/conveyor_belt/requirements.txt

# Install this package (editable)
uv pip install -e ~/conveyor_belt
```

## Usage

### View environment in browser (viser)

If you are on SSH, you need port forwarding for the viser web viewer:

```bash
# On your LOCAL machine, open an SSH tunnel:
ssh -L 8080:localhost:8080 apple@<server-ip>
```

Then on the server, run the test script with `--viz viser`:

```bash
cd ~/IsaacLab && source vibhaenv/bin/activate
python ~/conveyor_belt/conveyor_belt/test_env.py --num_envs 1 --steps 100000 --viz viser
```

Open `http://localhost:8080/` in your local browser to view the scene.

### Train

```bash
cd ~/IsaacLab && source vibhaenv/bin/activate
python ~/conveyor_belt/conveyor_belt/train.py --num_envs 4096 --max_iterations 1500 --headless
```

### Monitor training

```bash
# On your LOCAL machine, also tunnel port 6006:
ssh -L 6006:localhost:6006 apple@<server-ip>

# On the server:
cd ~/conveyor_belt && tensorboard --logdir logs/conveyor_belt
```

Open `http://localhost:6006/` in your local browser.

### Record demos

```bash
cd ~/IsaacLab && source vibhaenv/bin/activate
python ~/conveyor_belt/conveyor_belt/record_video.py \
    --checkpoint ~/conveyor_belt/logs/conveyor_belt/<run>/model_<iter>.pt \
    --num_demos 10 --enable_cameras
```

Successful picks are saved as individual MP4s in `~/conveyor_belt/demos/`.

## File Structure

```
conveyor_belt/
├── __init__.py              # Gym registration
├── conveyor_env_cfg.py      # Top-level env config (actions, obs, rewards, events)
├── scene_cfg.py             # Scene (robot, belt, YCB objects, rails, legs)
├── mdp/
│   ├── observations.py      # Object positions, velocities, EE vectors, conveyor vel, target one-hot
│   ├── rewards.py           # Approach, lift, grasp-and-lift (target object only)
│   ├── terminations.py      # Out-of-bounds, success (target object only)
│   └── events.py            # Belt driving, object reset, conveyor randomization, target selection
├── agents/
│   └── rsl_rl_ppo_cfg.py   # PPO hyperparameters
├── train.py                 # Training script with periodic eval
├── test_env.py              # Smoke test
└── record_video.py          # Record successful demo videos
```
