"""Per-step safety signals and per-episode driving metrics.

Adds to ``info``:
  * ``safety``   – dict with ttc, headway, gap, acceleration, jerk, lane_change, rss_violation
  * ``cost``     – CMDP cost c_t used by PPO-Lagrangian
  * ``episode_metrics`` (only on the last step) – everything the evaluation harness reports
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from highway_env.vehicle.kinematics import Vehicle

INF = float("inf")
TTC_CAP = 99.0  # cap for "no conflict" so averages stay finite


def lead_object_state(env) -> tuple[float, float, float, bool]:
    """(bumper gap [m], ego speed, lead speed, lead is a vehicle) for the nearest object ahead in the ego lane.

    The lead can be a static obstacle (the barrier closing the on-ramp): it matters for time-to-collision,
    but car-following measures (headway, RSS) are only defined with respect to vehicles.
    """
    u = env.unwrapped
    ego = u.vehicle
    if ego.lane_index is None:
        return INF, ego.speed, 0.0, False
    front, _ = u.road.neighbour_vehicles(ego, ego.lane_index)
    if front is None:
        return INF, ego.speed, 0.0, False
    length = 0.5 * (ego.LENGTH + getattr(front, "LENGTH", ego.LENGTH))
    gap = ego.lane_distance_to(front) - length
    return gap, ego.speed, float(getattr(front, "speed", 0.0)), isinstance(front, Vehicle)


def lead_vehicle_state(env) -> tuple[float, float, float]:
    """(gap, ego speed, lead speed) to the object ahead (vehicle or obstacle); used for TTC checks."""
    gap, v_ego, v_lead, _ = lead_object_state(env)
    return gap, v_ego, v_lead


def time_to_collision(gap: float, v_ego: float, v_lead: float) -> float:
    if gap <= 0:
        return 0.0
    closing = v_ego - v_lead
    return gap / closing if closing > 1e-6 else INF


def time_headway(gap: float, v_ego: float) -> float:
    if gap <= 0:
        return 0.0
    return gap / v_ego if v_ego > 0.1 else INF


def rss_safe_distance(
    v_rear: float,
    v_front: float,
    response_time: float = 0.5,
    accel_max: float = 3.0,
    brake_min: float = 4.0,
    brake_max: float = 8.0,
) -> float:
    """Longitudinal RSS safe distance (Shalev-Shwartz et al., 2017, Def. 1)."""
    v_resp = v_rear + response_time * accel_max
    d = (
        v_rear * response_time
        + 0.5 * accel_max * response_time**2
        + v_resp**2 / (2 * brake_min)
        - v_front**2 / (2 * brake_max)
    )
    return max(0.0, d)


class SafetyMonitor(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        ttc_threshold: float = 2.0,
        headway_threshold: float = 1.0,
        crash_cost: float = 10.0,
        near_miss_ttc: float = 1.0,
        rss: dict | None = None,
    ):
        super().__init__(env)
        self.ttc_threshold = ttc_threshold
        self.headway_threshold = headway_threshold
        self.crash_cost = crash_cost
        self.near_miss_ttc = near_miss_ttc
        self.rss = rss or {}

    # ------------------------------------------------------------------ utils
    def _dt(self) -> float:
        return 1.0 / self.env.unwrapped.config["policy_frequency"]

    def _measure(self) -> dict:
        gap, v_ego, v_lead, is_vehicle = lead_object_state(self.env)
        ttc = time_to_collision(gap, v_ego, v_lead)  # also towards static obstacles
        headway = time_headway(gap, v_ego) if is_vehicle else INF
        rss_violation = is_vehicle and gap < rss_safe_distance(v_ego, v_lead, **self.rss)
        return {
            "gap": min(gap, 1e3),
            "ttc": min(ttc, TTC_CAP),
            "headway": min(headway, TTC_CAP),
            "rss_violation": bool(rss_violation),
        }

    # -------------------------------------------------------------- gym API
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        ego = self.env.unwrapped.vehicle
        self._prev_speed = ego.speed
        self._prev_acc = None  # no jerk on the first step
        self._prev_lane = ego.lane_index
        self._ep = {
            "steps": 0,
            "return": 0.0,
            "cost": 0.0,
            "speed_sum": 0.0,
            "distance": 0.0,
            "min_ttc": TTC_CAP,
            "min_headway": TTC_CAP,
            "ttc_below_1_5": 0,
            "headway_below_1": 0,
            "jerk_sq": 0.0,
            "acc_sq": 0.0,
            "lane_changes": 0,
            "rss_violations": 0,
            "shield_steps": 0,
        }
        m = self._measure()
        info["safety"] = {**m, "acceleration": 0.0, "jerk": 0.0, "lane_change": False}
        info["cost"] = 0.0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        ego = self.env.unwrapped.vehicle
        dt = self._dt()

        m = self._measure()
        acc = (ego.speed - self._prev_speed) / dt
        jerk = 0.0 if self._prev_acc is None else (acc - self._prev_acc) / dt
        lane = ego.lane_index
        lane_change = bool(
            lane is not None
            and self._prev_lane is not None
            and lane[:2] == self._prev_lane[:2]
            and lane[2] != self._prev_lane[2]
        )
        self._prev_speed, self._prev_acc, self._prev_lane = ego.speed, acc, lane

        crashed = bool(info.get("crashed", ego.crashed))
        cost = (
            float(m["ttc"] < self.ttc_threshold)
            + float(m["headway"] < self.headway_threshold)
            + self.crash_cost * float(crashed)
        )
        info["safety"] = {**m, "acceleration": acc, "jerk": jerk, "lane_change": lane_change}
        info["cost"] = cost

        ep = self._ep
        ep["steps"] += 1
        ep["return"] += float(reward)
        ep["cost"] += cost
        ep["speed_sum"] += ego.speed
        ep["distance"] += ego.speed * dt
        ep["min_ttc"] = min(ep["min_ttc"], m["ttc"])
        ep["min_headway"] = min(ep["min_headway"], m["headway"])
        ep["ttc_below_1_5"] += int(m["ttc"] < 1.5)
        ep["headway_below_1"] += int(m["headway"] < 1.0)
        ep["jerk_sq"] += jerk**2
        ep["acc_sq"] += acc**2
        ep["lane_changes"] += int(lane_change)
        ep["rss_violations"] += int(m["rss_violation"])
        ep["shield_steps"] += int(info.get("shield_active", False))

        if terminated or truncated:
            info["episode_metrics"] = self._summarise(info, crashed)
        return obs, reward, terminated, truncated, info

    def _summarise(self, info: dict, crashed: bool) -> dict:
        ep = self._ep
        n = max(ep["steps"], 1)
        km = max(ep["distance"] / 1000.0, 1e-6)
        success = bool(info["is_success"]) if "is_success" in info else not crashed
        return {
            "crashed": float(crashed),
            "success": float(success),
            "merged": float(info.get("merged", success)),
            "return": ep["return"],
            "cost": ep["cost"],
            "length": ep["steps"],
            "avg_speed": ep["speed_sum"] / n,
            "distance": ep["distance"],
            "min_ttc": ep["min_ttc"],
            "min_headway": ep["min_headway"],
            "near_miss": float((ep["min_ttc"] < self.near_miss_ttc) and not crashed),
            "ttc_below_1_5_frac": ep["ttc_below_1_5"] / n,
            "headway_below_1_frac": ep["headway_below_1"] / n,
            "rms_jerk": float(np.sqrt(ep["jerk_sq"] / n)),
            "rms_acc": float(np.sqrt(ep["acc_sq"] / n)),
            "lane_changes": ep["lane_changes"],
            "lane_changes_per_km": ep["lane_changes"] / km,
            "rss_violation_frac": ep["rss_violations"] / n,
            "shield_frac": ep["shield_steps"] / n,
        }
