"""Register our custom scenarios with gymnasium."""

import gymnasium as gym


def register_envs() -> None:
    if "RampMerge-v0" in gym.registry:
        return
    gym.register(
        id="RampMerge-v0",
        entry_point="safelane.envs.ramp_merge:RampMergeEnv",
    )
