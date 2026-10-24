"""Privileged 'oracle' planner used ONLY as a benchmark / sanity check.

It reads the true world layout (people, fires, planning grid), follows the BFS
distance field to the nearest goal and drives with pure pursuit. A learned
policy only sees lidar + camera-style detections, so it is not expected to
match this at first; it tells us the task is solvable in the physics sim.
"""
from __future__ import annotations

import math

import numpy as np


class OraclePlanner:
    def __init__(self, env):
        self.env = env
        self._key = None
        self._dist = None
        self._stuck = 0
        self._recover = 0
        self._last_steer = 0.0

    def _goal(self):
        e = self.env
        pos, _, _ = e.robot.pose()
        people = [(math.hypot(x - pos[0], y - pos[1]), ("p", i), (x, y))
                  for i, (x, y) in enumerate(e.layout.people) if not e.located[i]]
        if people:
            return min(people)
        fires = [(math.hypot(x - pos[0], y - pos[1]), ("f", i), (x, y))
                 for i, (x, y) in enumerate(e.layout.fires) if e.fire_int[i] > 0]
        return min(fires) if fires else (None, None, None)

    def act(self):
        a = self._plan()
        vx, _, _ = self.env.robot.body_velocity()
        if self._recover > 0:                       # back out, steering the opposite way
            self._recover -= 1
            return np.array([-0.6, -self._last_steer, 0.0], np.float32)
        self._stuck = self._stuck + 1 if (abs(vx) < 0.08 and a[0] > 0.2) else 0
        if self._stuck > 15:
            self._stuck, self._recover = 0, 30
        self._last_steer = float(a[1]) if abs(a[1]) > 0.2 else 1.0
        return a

    def _plan(self):
        e = self.env
        g = e.layout.grid
        pos, yaw, _ = e.robot.pose()
        d, key, tgt = self._goal()
        if key is None:
            return np.zeros(3, np.float32)
        if key != self._key:
            # goal cells: reachable free cells within the stopping radius of the target
            stop = 1.4 if key[0] == "p" else 2.0
            def los(ix, iy):
                x0, y0 = g.to_xy(ix, iy)
                n = max(2, int(math.hypot(tgt[0] - x0, tgt[1] - y0) / (g.cell * 0.5)))
                for k in range(1, n):
                    qx, qy = g.to_cell(x0 + (tgt[0] - x0) * k / n, y0 + (tgt[1] - y0) * k / n)
                    if g.occ[qx, qy]:
                        return False
                return True
            near = [(ix, iy) for ix in range(g.n) for iy in range(g.n)
                    if not g.inflated[ix, iy] and math.hypot(*(np.array(g.to_xy(ix, iy)) - tgt)) <= stop]
            cells = [c for c in near if los(*c)] or near
            self._dist = g.bfs(cells) if cells else None
            self._key = key
        pump = 0.0
        if key[0] == "f":
            _, _, _ = pos, yaw, tgt
            bearing = (math.atan2(tgt[1] - pos[1], tgt[0] - pos[0]) - yaw + math.pi) % (2 * math.pi) - math.pi
            if d <= 2.2 and abs(bearing) < 0.4:
                pump = 1.0
        if self._dist is None:
            return np.zeros(3, np.float32)
        ix, iy = g.to_cell(pos[0], pos[1])
        # look ahead ~0.9 m along the descending BFS field
        cx, cy = ix, iy
        for _ in range(int(0.9 / g.cell)):
            best, bv = None, self._dist[cx, cy] if self._dist[cx, cy] >= 0 else 10 ** 9
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < g.n and 0 <= ny < g.n and 0 <= self._dist[nx, ny] < bv:
                        best, bv = (nx, ny), self._dist[nx, ny]
            if best is None:
                break
            cx, cy = best
        wx, wy = g.to_xy(cx, cy)
        if self._dist[cx, cy] == 0 and key[0] == "f":
            wx, wy = tgt                          # at the stop ring: turn to face the fire
        err = (math.atan2(wy - pos[1], wx - pos[0]) - yaw + math.pi) % (2 * math.pi) - math.pi
        steer = float(np.clip(1.8 * err, -1, 1))
        speed = 0.7 * max(0.25, math.cos(err)) if abs(err) < 1.2 else 0.25
        if pump > 0:
            speed = 0.0
            steer = float(np.clip(1.8 * err, -1, 1))
        return np.array([speed, steer, pump], np.float32)
