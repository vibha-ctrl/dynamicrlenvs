"""Conveyor-belt manipulation environment for Isaac Lab.

Importing this package registers the Gymnasium environment so it can be
created with ``gymnasium.make("Isaac-ConveyorBelt-Franka-v0")``.
"""

import gymnasium as gym

gym.register(
    id="Isaac-ConveyorBelt-Franka-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.conveyor_env_cfg:ConveyorBeltEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:ConveyorBeltPPORunnerCfg",
    },
)
