# Copyright (c) 2024-2026. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL PPO configuration for the conveyor-belt manipulation task."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class ConveyorBeltPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """On-policy PPO runner tuned for the conveyor-belt pick task."""

    seed: int = 42
    num_steps_per_env: int = 24
    max_iterations: int = 1500
    save_interval: int = 100
    experiment_name: str = "conveyor_belt"
    run_name: str = "ppo"
    logger: str = "tensorboard"

    policy: RslRlPpoActorCriticCfg = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )

    algorithm: RslRlPpoAlgorithmCfg = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
