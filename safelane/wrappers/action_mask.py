"""Safety shield over DiscreteMetaAction (ablation only — the RL policy still decides).

A meta-action is masked when it is road-illegal (highway-env's own availability check)
or when it would create a short time-to-collision:
  * LANE_LEFT / LANE_RIGHT: gap or TTC to the leader / follower in the target lane too small
  * FASTER: TTC or headway to the current leader already below threshold
IDLE and SLOWER are never masked for safety reasons, so the mask is never empty.

``action_masks()`` is what sb3-contrib's MaskablePPO queries. With ``enforce=True`` the
wrapper also replaces an unsafe chosen action (post-posed shield for non-maskable agents).
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from safelane.wrappers.safety import lead_vehicle_state, time_headway, time_to_collision


class SafetyShield(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        ttc_threshold: float = 2.5,
        min_gap: float = 8.0,
        headway_threshold: float = 0.8,
        enforce: bool = False,
    ):
        super().__init__(env)
        self.ttc_threshold = ttc_threshold
        self.min_gap = min_gap
        self.headway_threshold = headway_threshold
        self.enforce = enforce

    def _lane_change_safe(self, lane_index) -> bool:
        u = self.env.unwrapped
        ego = u.vehicle
        front, rear = u.road.neighbour_vehicles(ego, lane_index)
        if front is not None:
            gap = ego.lane_distance_to(front) - ego.LENGTH
            if gap < self.min_gap or time_to_collision(gap, ego.speed, front.speed) < self.ttc_threshold:
                return False
        if rear is not None:
            gap = -ego.lane_distance_to(rear) - ego.LENGTH
            if gap < self.min_gap or time_to_collision(gap, rear.speed, ego.speed) < self.ttc_threshold:
                return False
        return True

    def legal_mask(self) -> np.ndarray:
        u = self.env.unwrapped
        mask = np.zeros(u.action_space.n, dtype=bool)
        mask[u.action_type.get_available_actions()] = True
        return mask

    def action_masks(self) -> np.ndarray:
        u = self.env.unwrapped
        ego = u.vehicle
        idx = u.action_type.actions_indexes
        mask = self.legal_mask()
        network = u.road.network
        if ego.lane_index is not None:
            for side in network.side_lanes(ego.lane_index):
                key = "LANE_LEFT" if side[2] < ego.lane_index[2] else "LANE_RIGHT"
                if key in idx and mask[idx[key]] and not self._lane_change_safe(side):
                    mask[idx[key]] = False
        if "FASTER" in idx and mask[idx["FASTER"]]:
            gap, v_ego, v_lead = lead_vehicle_state(self.env)
            if (
                time_to_collision(gap, v_ego, v_lead) < self.ttc_threshold
                or time_headway(gap, v_ego) < self.headway_threshold
            ):
                mask[idx["FASTER"]] = False
        if not mask.any():
            mask[idx["IDLE"]] = True
        return mask

    def _fallback(self, mask: np.ndarray) -> int:
        idx = self.env.unwrapped.action_type.actions_indexes
        for name in ("SLOWER", "IDLE"):
            if name in idx and mask[idx[name]]:
                return idx[name]
        return int(np.flatnonzero(mask)[0])

    def step(self, action):
        mask = self.action_masks()
        shield_active = bool((self.legal_mask() & ~mask).any())
        intervened = False
        if self.enforce and not mask[int(action)]:
            action = self._fallback(mask)
            intervened = True
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["shield_active"] = shield_active
        info["shield_intervened"] = intervened
        return obs, reward, terminated, truncated, info
