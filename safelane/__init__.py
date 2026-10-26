"""SafeLane-RL: safe, robust and explainable RL for highway driving."""

import highway_env  # noqa: F401  (registers the stock highway-env scenarios)

from safelane.envs.fast_observation import install as _install_fast_observation
from safelane.envs.registration import register_envs

register_envs()
_install_fast_observation()

__version__ = "1.0.0"
