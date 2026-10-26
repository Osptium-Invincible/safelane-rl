"""Heterogeneous traffic models used for domain randomisation.

Every vehicle samples its own IDM/MOBIL parameters at creation time, so a single
episode contains a mix of timid and pushy drivers. The class-level ranges define the
population; instance attributes shadow the class constants that IDMVehicle reads
through ``self.<NAME>``.
"""

from __future__ import annotations

from highway_env.vehicle.behavior import IDMVehicle


class RandomizedIDMVehicle(IDMVehicle):
    """IDM + MOBIL driver with per-vehicle random parameters (moderate population)."""

    TIME_WANTED_RANGE = (1.0, 2.0)  # desired time gap T [s]
    COMFORT_ACC_RANGE = (2.0, 4.0)  # a_max comfort [m/s2]
    DISTANCE_WANTED_RANGE = (7.0, 12.0)  # jam distance s0 (incl. length) [m]
    POLITENESS_RANGE = (0.0, 0.5)  # MOBIL politeness p
    ACC_GAIN_RANGE = (0.1, 0.4)  # MOBIL lane-change threshold [m/s2]
    SPEED_FACTOR_RANGE = (0.9, 1.1)  # multiplier on the desired speed

    def __init__(self, road, position, *args, **kwargs):
        super().__init__(road, position, *args, **kwargs)
        self._sample_parameters(scale_speed=True)

    def _sample_parameters(self, scale_speed: bool) -> None:
        rng = self.road.np_random
        self.TIME_WANTED = rng.uniform(*self.TIME_WANTED_RANGE)
        self.COMFORT_ACC_MAX = rng.uniform(*self.COMFORT_ACC_RANGE)
        self.DISTANCE_WANTED = rng.uniform(*self.DISTANCE_WANTED_RANGE)
        self.POLITENESS = rng.uniform(*self.POLITENESS_RANGE)
        self.LANE_CHANGE_MIN_ACC_GAIN = rng.uniform(*self.ACC_GAIN_RANGE)
        self.DELTA = rng.uniform(*self.DELTA_RANGE)
        if scale_speed and self.target_speed is not None:
            self.target_speed *= rng.uniform(*self.SPEED_FACTOR_RANGE)

    def randomize_behavior(self):
        # highway-env calls this right after construction: resample the driving style only, so the
        # desired-speed factor is applied exactly once (in __init__)
        self._sample_parameters(scale_speed=False)


class AggressiveIDMVehicle(RandomizedIDMVehicle):
    """Short gaps, impolite and eager lane changes, faster than the flow."""

    TIME_WANTED_RANGE = (0.5, 1.0)
    COMFORT_ACC_RANGE = (3.5, 5.0)
    DISTANCE_WANTED_RANGE = (6.0, 8.0)
    POLITENESS_RANGE = (0.0, 0.1)
    ACC_GAIN_RANGE = (0.0, 0.1)
    SPEED_FACTOR_RANGE = (1.05, 1.2)


class CautiousIDMVehicle(RandomizedIDMVehicle):
    """Long gaps, polite, reluctant lane changes, slower than the flow."""

    TIME_WANTED_RANGE = (1.8, 2.5)
    COMFORT_ACC_RANGE = (1.5, 2.5)
    DISTANCE_WANTED_RANGE = (10.0, 15.0)
    POLITENESS_RANGE = (0.4, 1.0)
    ACC_GAIN_RANGE = (0.4, 0.8)
    SPEED_FACTOR_RANGE = (0.8, 0.95)


TRAFFIC_MODELS = {
    "idm": "highway_env.vehicle.behavior.IDMVehicle",
    "randomized": "safelane.envs.vehicles.RandomizedIDMVehicle",
    "aggressive": "safelane.envs.vehicles.AggressiveIDMVehicle",
    "cautious": "safelane.envs.vehicles.CautiousIDMVehicle",
    "linear_aggressive": "highway_env.vehicle.behavior.AggressiveVehicle",
    "linear_defensive": "highway_env.vehicle.behavior.DefensiveVehicle",
}


def resolve_vehicle_type(name: str) -> str:
    """Accept either a short alias from TRAFFIC_MODELS or a full dotted path."""
    return TRAFFIC_MODELS.get(name, name)
