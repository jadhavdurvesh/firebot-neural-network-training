"""Mission memory: known people/fires, located flags and coverage map.

This mirrors the state the simulator tracks internally, so the observation the
policy sees on the robot is built from the same quantities as in training.
"""
from __future__ import annotations

import math

import numpy as np

from firebot_sim import obs as obsmod
from .hal import SensorFrame


class MissionMemory:
    def __init__(self, n_people: int, n_fires: int, arena_half: float, locate_dist: float = 1.8,
                 coverage_grid: int = 6, merge_dist: float = 0.8):
        self.n_people, self.n_fires = n_people, n_fires
        self.A, self.locate_dist, self.merge = arena_half, locate_dist, merge_dist
        self.cov_n = coverage_grid
        self.reset()

    def reset(self):
        self.people = []      # dicts: x, y, located
        self.fires = []       # dicts: x, y, active
        self.visited = np.zeros((self.cov_n, self.cov_n), bool)

    @staticmethod
    def _match(items, x, y, tol):
        for it in items:
            if math.hypot(it["x"] - x, it["y"] - y) <= tol:
                return it
        return None

    def update(self, f: SensorFrame):
        visible_people = []
        for d in f.detections:
            if d.cls == "person":
                it = self._match(self.people, d.x, d.y, self.merge)
                if it is None:
                    it = {"x": d.x, "y": d.y, "located": False}
                    self.people.append(it)
                visible_people.append(it)
            elif d.cls == "fire":
                it = self._match(self.fires, d.x, d.y, self.merge)
                if it is None:
                    it = {"x": d.x, "y": d.y, "active": True}
                    self.fires.append(it)
                it["active"] = bool(d.active)
        for it in self.people:
            if it["located"]:
                continue
            dist = math.hypot(it["x"] - f.x, it["y"] - f.y)
            if dist <= self.locate_dist and (any(it is v for v in visible_people) or dist < 1.0):
                it["located"] = True
        obsmod.update_coverage(self.visited, f.x, f.y, self.A)

    @property
    def located_count(self):
        return sum(p["located"] for p in self.people)

    @property
    def all_people_located(self):
        return self.located_count >= self.n_people

    @property
    def any_fire_active(self):
        out = sum(1 for fi in self.fires if not fi["active"])
        return out < self.n_fires
