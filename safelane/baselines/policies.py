"""Non-learning reference drivers (comparison baselines only — never used for training).

Every policy has the signature ``policy(obs, env) -> int`` where ``env`` is the
(wrapped) gymnasium env, so rule-based drivers can read the simulator state.
"""

from __future__ import annotations

import numpy as np
from highway_env.vehicle.behavior import IDMVehicle
from highway_env.vehicle.objects import Obstacle

RAMP_MERGE_LANE = ("b", "c", 2)
RAMP_APPROACH_EDGES = {("j", "k"), ("k", "b")}
MAIN_EDGES_BEFORE_MERGE_END = {("a", "b"), ("b", "c")}


class RandomPolicy:
    name = "random"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def __call__(self, obs, env) -> int:
        return int(self.rng.integers(env.action_space.n))


class ConstantSpeedPolicy:
    """Always IDLE: keep lane and current target speed."""

    name = "constant_speed"

    def __call__(self, obs, env) -> int:
        return int(env.unwrapped.action_type.actions_indexes["IDLE"])


class IDMMobilPolicy:
    """The classic rule-based AV stack mapped onto meta-actions.

    * Longitudinal: IDM acceleration towards the top allowed speed -> FASTER / IDLE / SLOWER.
    * Lateral: MOBIL lane-change criterion on the highway.
    * On-ramp: classic traffic-engineering merge logic. On the ramp, match the speed of the target
      lane; in the merge zone, merge when the gaps ahead and behind exceed a critical time gap
      (gap acceptance), otherwise adapt speed to line up with the gap behind or ahead.
    """

    name = "idm_mobil"

    def __init__(
        self,
        accel_threshold: float = 0.5,
        brake_threshold: float = -1.0,
        critical_gap_time: float = 0.8,
        min_gap: float = 4.0,
        forced_merge_distance: float = 25.0,
        obstacle_lookahead: float = 25.0,
    ):
        self.accel_threshold = accel_threshold
        self.brake_threshold = brake_threshold
        self.critical_gap_time = critical_gap_time
        self.min_gap = min_gap
        self.forced_merge_distance = forced_merge_distance
        self.obstacle_lookahead = obstacle_lookahead
        self._gap_seek = None
        self._seek_dir = None

    def __call__(self, obs, env) -> int:
        u = env.unwrapped
        ego, road = u.vehicle, u.road
        idx = u.action_type.actions_indexes
        available = set(u.action_type.get_available_actions())

        shadow = IDMVehicle.create_from(ego)
        shadow.target_speed = float(np.max(ego.target_speeds))
        slot = road.vehicles.index(ego)
        road.vehicles[slot] = shadow  # let MOBIL see the world from the ego's position
        try:
            lane_action = self._lane_decision(shadow, ego, road, idx, available)
            front, _ = road.neighbour_vehicles(shadow, shadow.lane_index)
            if (
                isinstance(front, Obstacle)
                and shadow.lane_distance_to(front) > self.obstacle_lookahead
            ):
                front = None  # the ramp-end barrier is not a slow leader: keep speed to merge
            acc = shadow.acceleration(shadow, front_vehicle=front)
        finally:
            road.vehicles[slot] = ego

        if lane_action is not None:
            return lane_action
        if self._gap_seek is not None and self._gap_seek in available:
            return self._gap_seek
        if acc < self.brake_threshold and idx["SLOWER"] in available:
            return idx["SLOWER"]
        if acc > self.accel_threshold and idx["FASTER"] in available:
            return idx["FASTER"]
        return idx["IDLE"]

    def _lane_decision(self, shadow, ego, road, idx, available):
        self._gap_seek = None
        if ego.lane_index != RAMP_MERGE_LANE:
            self._seek_dir = None  # new episode / merged: forget the gap-seeking commitment
        if ego.lane_index[:2] in RAMP_APPROACH_EDGES:
            self._gap_seek = self._match_target_lane_speed(ego, road, idx)
            return None
        if ego.lane_index != ego.target_lane_index:
            return None  # lane change in progress
        for lane_index in road.network.side_lanes(ego.lane_index):
            if not road.network.get_lane(lane_index).is_reachable_from(ego.position):
                continue
            key = "LANE_LEFT" if lane_index[2] < ego.lane_index[2] else "LANE_RIGHT"
            if idx.get(key) not in available:
                continue
            if ego.lane_index == RAMP_MERGE_LANE:  # mandatory merge
                merge, adjust, can_speed_up = self._gap_acceptance(shadow, ego, road, lane_index)
                if merge:
                    return idx[key]
                # commit to one direction (alternating faster/slower never lines up with a gap);
                # when pulling ahead and already faster than the leader, hold speed instead of reversing
                self._seek_dir = self._seek_dir or adjust
                if self._seek_dir == "FASTER" and (not can_speed_up or idx["FASTER"] not in available):
                    self._gap_seek = idx["IDLE"]
                else:
                    self._gap_seek = idx[self._seek_dir]
            elif shadow.mobil(lane_index):
                return idx[key]
        return None

    def _match_target_lane_speed(self, ego, road, idx):
        """On the ramp: accelerate/brake towards the mean speed of the main-road lane we will join."""
        speeds = [v.speed for v in road.vehicles
                  if v is not ego and v.lane_index is not None and v.lane_index[2] == 1
                  and v.lane_index[:2] in MAIN_EDGES_BEFORE_MERGE_END]
        if not speeds:
            return None
        target = float(np.mean(speeds))
        if ego.speed < target - 2.5:
            return idx["FASTER"]
        if ego.speed > target + 2.5:
            return idx["SLOWER"]
        return idx["IDLE"]

    def _gap_acceptance(self, shadow, ego, road, lane_index) -> tuple[bool, str, bool]:
        """Accept the gap if lead and lag gaps exceed the critical time gap (+ closing-speed margin)."""
        front, rear = road.neighbour_vehicles(shadow, lane_index)
        v = ego.speed
        lead_gap = shadow.lane_distance_to(front) - ego.LENGTH if front is not None else np.inf
        lag_gap = -shadow.lane_distance_to(rear) - ego.LENGTH if rear is not None else np.inf
        v_lead = front.speed if front is not None else v
        v_lag = rear.speed if rear is not None else v
        # critical gaps shrink when the relative speed opens the gap (faster than the lag car, slower than the lead)
        need_lead = max(self.min_gap, self.critical_gap_time * v + (v - v_lead))
        need_lag = max(self.min_gap, self.critical_gap_time * v_lag + (v_lag - v))
        lane = road.network.get_lane(RAMP_MERGE_LANE)
        remaining = lane.length - lane.local_coordinates(ego.position)[0]
        if remaining < self.forced_merge_distance:  # end of the ramp: forced merge with minimal gaps
            need_lead, need_lag = self.min_gap, self.min_gap
        can_speed_up = v < v_lead + 3.0
        if lead_gap >= need_lead and lag_gap >= need_lag:
            return True, "IDLE", can_speed_up
        # line up with a gap: pull ahead of the lag car if there is room in front, else drop behind it
        if lag_gap < need_lag and lead_gap > need_lead + 5.0 and can_speed_up:
            return False, "FASTER", can_speed_up
        return False, "SLOWER", can_speed_up


BASELINES = {
    "random": RandomPolicy,
    "constant_speed": ConstantSpeedPolicy,
    "idm_mobil": IDMMobilPolicy,
}
