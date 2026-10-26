"""RampMerge-v0: the ego vehicle starts on the on-ramp and must merge into traffic.

highway-env's stock ``merge-v0/v1`` puts the ego on the main road (the merging car
is an NPC), so it cannot measure the ego's own merging skill. This scenario reuses
the merge road geometry but spawns the ego on the ramp and exposes an explicit
``info["is_success"]`` flag.

Road layout (x in metres, from MergeEnv): main road lanes 0/1 run 0 -> 460;
the ramp runs straight (0-150), converges (150-230), then runs parallel to lane 1
(230-310) where merging is possible; an obstacle closes the ramp at x = 310.
"""

from __future__ import annotations

import numpy as np
from highway_env import utils
from highway_env.envs.merge_env import ConnectedLaneMergeEnv

from safelane.envs.geometric_road import GeometricRoad

MAIN_ROAD_EDGES = {("b", "c"), ("c", "d")}
RAMP_LANE_IN_MERGE_ZONE = ("b", "c", 2)


class RampMergeEnv(ConnectedLaneMergeEnv):
    """On-ramp merging with configurable main-road traffic density."""

    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update(
            {
                "action": {
                    "type": "DiscreteMetaAction",
                    "target_speeds": [10, 15, 20, 25, 30],
                },
                "duration": 25,  # [s] truncation
                "end_position": 370.0,  # [m] episode ends successfully past this x
                "vehicles_density": 1.0,  # main-road traffic density multiplier
                "max_vehicles": 24,
                "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
                "ego_spawn_range": [80.0, 120.0],  # [m] along the ramp
                "ego_speed_range": [20.0, 25.0],  # [m/s]
                "traffic_speed_range": [23.0, 29.0],  # [m/s]
                "collision_reward": -1.0,
                "high_speed_reward": 0.2,
                "merged_reward": 1.0,
                "ramp_reward": 0.0,  # optional per-step penalty while still on the ramp
                "reward_speed_range": [20, 30],
            }
        )
        return cfg

    # ------------------------------------------------------------------ scene
    def _reset(self) -> None:
        self._make_road()
        stock = self.road
        self.road = GeometricRoad(
            network=stock.network,
            road_objects=stock.objects,
            np_random=stock.np_random,
            record_history=stock.record_history,
            neighbour_vehicles_connected_lanes=self.config["neighbour_vehicles_connected_lanes"],
        )
        for obj in self.road.objects:
            obj.road = self.road
        self._make_vehicles()
        self._merged = False
        self._merged_now = False

    def _make_vehicles(self) -> None:
        road = self.road
        rng = self.np_random
        other_cls = utils.class_from_path(self.config["other_vehicles_type"])

        ramp = road.network.get_lane(("j", "k", 0))
        s0 = rng.uniform(*self.config["ego_spawn_range"])
        ego = self.action_type.vehicle_class(
            road,
            ramp.position(s0, 0.0),
            heading=ramp.heading_at(s0),
            speed=rng.uniform(*self.config["ego_speed_range"]),
        )
        road.vehicles.append(ego)

        density = max(float(self.config["vehicles_density"]), 1e-3)
        count = 0
        for lane_id in (0, 1):
            lane = road.network.get_lane(("a", "b", lane_id))
            x = rng.uniform(0.0, 20.0)
            while x < 220.0 and count < self.config["max_vehicles"]:
                v = other_cls(
                    road,
                    lane.position(x, 0.0),
                    heading=lane.heading_at(x),
                    speed=rng.uniform(*self.config["traffic_speed_range"]),
                )
                road.vehicles.append(v)  # target speed = spawn speed x the driver's own speed factor
                count += 1
                x += max(12.0, rng.uniform(35.0, 65.0) / density)
        self.vehicle = ego

    # --------------------------------------------------------------- dynamics
    def _on_main_road(self) -> bool:
        idx = self.vehicle.lane_index
        return idx is not None and idx[:2] in MAIN_ROAD_EDGES and idx != RAMP_LANE_IN_MERGE_ZONE

    def _simulate(self, action=None) -> None:
        super()._simulate(action)
        on_main = self._on_main_road() and not self.vehicle.crashed
        self._merged_now = on_main and not self._merged
        self._merged = self._merged or on_main

    # ----------------------------------------------------------------- reward
    def _rewards(self, action) -> dict[str, float]:
        forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
        scaled = utils.lmap(forward_speed, self.config["reward_speed_range"], [0, 1])
        return {
            "collision_reward": float(self.vehicle.crashed),
            "high_speed_reward": float(np.clip(scaled, 0, 1)),
            "merged_reward": float(self._merged_now),
            "ramp_reward": float(not self._merged),
        }

    def _reward(self, action) -> float:
        return float(
            sum(self.config.get(name, 0.0) * value for name, value in self._rewards(action).items())
        )

    # ------------------------------------------------------------ termination
    def _passed_end(self) -> bool:
        return bool(self.vehicle.position[0] > self.config["end_position"])

    def _is_terminated(self) -> bool:
        return bool(self.vehicle.crashed or self._passed_end())

    def _is_truncated(self) -> bool:
        return bool(self.time >= self.config["duration"])

    def _info(self, obs, action=None) -> dict:
        info = super()._info(obs, action)
        merged = getattr(self, "_merged", False)
        info["merged"] = bool(merged)
        info["is_success"] = bool(merged and not self.vehicle.crashed and self._passed_end())
        return info
