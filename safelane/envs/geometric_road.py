"""Road whose connected-lane neighbour search matches segments by geometry.

highway-env 1.12's ``neighbour_vehicles_connected_lanes`` links consecutive segments by
*lane id*. On merge roads the ramp is lane 0 on edges j->k and k->b but lane 2 on
b->c, so a ramp vehicle "sees" main-road lane-0 traffic as its leader, and main-road
lane 1 treats the ramp as its rear segment. This subclass links a lane to the lane of
the next / previous segment whose start / end point coincides with it.
"""

from __future__ import annotations

import numpy as np
from highway_env.road.road import Road
from highway_env.vehicle.objects import Landmark

CONNECT_TOL = 1.0  # [m] max distance between lane end and next lane start


class GeometricRoad(Road):
    def _connected_lanes(self, lane_index) -> list:
        _from, _to, _id = lane_index
        lane = self.network.get_lane(lane_index)
        end, start = lane.position(lane.length, 0), lane.position(0, 0)
        out = [(lane, 0.0)]
        for lanes in self.network.graph.get(_to, {}).values():
            dists = [np.linalg.norm(nl.position(0, 0) - end) for nl in lanes]
            k = int(np.argmin(dists))
            if dists[k] < CONNECT_TOL:
                out.append((lanes[k], lane.length))
        for to_dict in self.network.graph.values():
            if _from not in to_dict:
                continue
            lanes = to_dict[_from]
            dists = [np.linalg.norm(pl.position(pl.length, 0) - start) for pl in lanes]
            k = int(np.argmin(dists))
            if dists[k] < CONNECT_TOL:
                out.append((lanes[k], -lanes[k].length))
        return out

    def neighbour_vehicles(self, vehicle, lane_index=None):
        if not self.neighbour_vehicles_connected_lanes:
            return super().neighbour_vehicles(vehicle, lane_index)
        lane_index = lane_index or vehicle.lane_index
        if not lane_index:
            return None, None
        lane = self.network.get_lane(lane_index)
        s = lane.local_coordinates(vehicle.position)[0]
        s_front = s_rear = v_front = v_rear = None
        searched = self._connected_lanes(lane_index)
        for v in self.vehicles + self.objects:
            if v is vehicle or isinstance(v, Landmark):
                continue
            for search_lane, offset in searched:
                s_v, lat_v = search_lane.local_coordinates(v.position)
                if not search_lane.on_lane(v.position, s_v, lat_v, margin=1):
                    continue
                s_v += offset
                if s <= s_v and (s_front is None or s_v <= s_front):
                    s_front, v_front = s_v, v
                if s_v < s and (s_rear is None or s_v > s_rear):
                    s_rear, v_rear = s_v, v
                break
        return v_front, v_rear
