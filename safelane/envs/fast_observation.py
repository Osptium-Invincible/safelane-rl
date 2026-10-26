"""A numpy re-implementation of highway-env's Kinematics observation.

The stock ``KinematicObservation.observe`` builds pandas DataFrames on every step, which
takes ~20% of the whole simulation time. This subclass produces exactly the same
matrix (verified in tests/test_fast_observation.py) with plain numpy.

``install()`` patches the factory used by ``AbstractEnv.define_spaces`` so every
``{"type": "Kinematics"}`` observation uses it. Set ``SAFELANE_FAST_OBS=0`` to disable.
"""

from __future__ import annotations

import os

import numpy as np
from highway_env import utils
from highway_env.envs.common import abstract
from highway_env.envs.common.observation import KinematicObservation, observation_factory
from highway_env.road.lane import AbstractLane
from highway_env.vehicle.kinematics import Vehicle

DIRECT_FEATURES = {"presence", "x", "y", "vx", "vy", "heading", "cos_h", "sin_h"}
RELATIVE = ("x", "y", "vx", "vy")


class FastKinematicObservation(KinematicObservation):
    def _row(self, v, origin) -> list[float]:
        if isinstance(v, Vehicle) and DIRECT_FEATURES.issuperset(self.features):
            d = {
                "presence": 1.0,
                "x": v.position[0],
                "y": v.position[1],
                "heading": v.heading,
                "cos_h": np.cos(v.heading),
                "sin_h": np.sin(v.heading),
            }
            d["vx"], d["vy"] = v.velocity
            if origin is not None:
                d["x"] -= origin.position[0]
                d["y"] -= origin.position[1]
                ovx, ovy = origin.velocity
                d["vx"] -= ovx
                d["vy"] -= ovy
        else:
            d = v.to_dict(origin, observe_intentions=self.observe_intentions)
        return [d.get(f, np.nan) for f in self.features]

    def _normalize(self, arr: np.ndarray) -> np.ndarray:
        if not self.features_range:
            side_lanes = self.env.road.network.all_side_lanes(self.observer_vehicle.lane_index)
            self.features_range = {
                "x": [-5.0 * Vehicle.MAX_SPEED, 5.0 * Vehicle.MAX_SPEED],
                "y": [-AbstractLane.DEFAULT_WIDTH * len(side_lanes), AbstractLane.DEFAULT_WIDTH * len(side_lanes)],
                "vx": [-2 * Vehicle.MAX_SPEED, 2 * Vehicle.MAX_SPEED],
                "vy": [-2 * Vehicle.MAX_SPEED, 2 * Vehicle.MAX_SPEED],
            }
        for feature, (lo, hi) in self.features_range.items():
            if feature in self.features:
                j = self.features.index(feature)
                arr[:, j] = utils.lmap(arr[:, j], [lo, hi], [-1, 1])
                if self.clip:
                    arr[:, j] = np.clip(arr[:, j], -1, 1)
        return arr

    def observe(self) -> np.ndarray:
        if not self.env.road:
            return np.zeros(self.space().shape)
        ego = self.observer_vehicle
        others = self.env.road.close_objects_to(
            ego,
            self.env.PERCEPTION_DISTANCE,
            count=self.vehicles_count - 1,
            see_behind=self.see_behind,
            sort=self.order == "sorted",
            vehicles_only=not self.include_obstacles,
        )
        origin = None if self.absolute else ego
        rows = [self._row(ego, None)]
        if others:
            rows += [self._row(v, origin) for v in others[-self.vehicles_count + 1 :]]
        arr = np.array(rows, dtype=np.float64)
        if self.normalize:
            arr = self._normalize(arr)
        if arr.shape[0] < self.vehicles_count:
            arr = np.vstack([arr, np.zeros((self.vehicles_count - arr.shape[0], len(self.features)))])
        if self.order == "shuffled":
            self.env.np_random.shuffle(arr[1:])
        return arr.astype(self.space().dtype)


def _factory(env, config):
    if config["type"] == "Kinematics":
        return FastKinematicObservation(env, **config)
    return observation_factory(env, config)


def install() -> None:
    if os.environ.get("SAFELANE_FAST_OBS", "1") != "0":
        abstract.observation_factory = _factory
