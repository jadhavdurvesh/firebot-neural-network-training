"""Random arena generation with guaranteed reachability.

A coarse occupancy grid (inflated by the robot radius) is used only for
(a) checking that every person/fire is reachable and (b) the privileged
oracle baseline. The learning agent never sees this grid.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from .config import WorldConfig


@dataclass
class Box:
    cx: float
    cy: float
    hx: float
    hy: float
    yaw: float = 0.0


@dataclass
class WorldLayout:
    boxes: List[Box]
    people: List[Tuple[float, float]]
    fires: List[Tuple[float, float]]
    spawn: Tuple[float, float, float]            # x, y, yaw
    grid: "GridMap" = None


class GridMap:
    def __init__(self, half: float, cell: float):
        self.half, self.cell = half, cell
        self.n = int(round(2 * half / cell))
        xs = (np.arange(self.n) + 0.5) * cell - half
        self.X, self.Y = np.meshgrid(xs, xs, indexing="ij")   # [ix, iy]
        self.occ = np.zeros((self.n, self.n), dtype=bool)
        self.inflated = None

    def add_box(self, b: Box):
        c, s = np.cos(b.yaw), np.sin(b.yaw)
        dx, dy = self.X - b.cx, self.Y - b.cy
        lx, ly = c * dx + s * dy, -s * dx + c * dy
        self.occ |= (np.abs(lx) <= b.hx) & (np.abs(ly) <= b.hy)

    def inflate(self, radius: float):
        r = int(np.ceil(radius / self.cell))
        out = self.occ.copy()
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                if dx * dx + dy * dy <= r * r:
                    out |= np.roll(np.roll(self.occ, dx, 0), dy, 1)
        # border band counts as blocked (boundary walls)
        out[:r, :] = out[-r:, :] = True
        out[:, :r] = out[:, -r:] = True
        self.inflated = out

    def to_cell(self, x, y):
        ix = int(np.clip((x + self.half) / self.cell, 0, self.n - 1))
        iy = int(np.clip((y + self.half) / self.cell, 0, self.n - 1))
        return ix, iy

    def to_xy(self, ix, iy):
        return (ix + 0.5) * self.cell - self.half, (iy + 0.5) * self.cell - self.half

    def bfs(self, start_cells):
        """8-connected BFS distance (in cells) over the inflated free space."""
        dist = np.full((self.n, self.n), -1, dtype=np.int32)
        dq = deque()
        for c in start_cells:
            if not self.inflated[c]:
                dist[c] = 0
                dq.append(c)
        while dq:
            x, y = dq.popleft()
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.n and 0 <= ny < self.n and dist[nx, ny] < 0 and not self.inflated[nx, ny]:
                        dist[nx, ny] = dist[x, y] + 1
                        dq.append((nx, ny))
        return dist


def _ri(rng, lo_hi):
    return int(rng.integers(lo_hi[0], lo_hi[1] + 1))


def generate_world(rng: np.random.Generator, wc: WorldConfig, robot_radius: float,
                   n_people=None, n_fires=None, max_tries: int = 200) -> WorldLayout:
    A = wc.arena_half
    for _ in range(max_tries):
        boxes: List[Box] = []
        for _ in range(_ri(rng, wc.n_walls)):
            length = rng.uniform(2.5, 5.0)
            horiz = rng.random() < 0.5
            cx, cy = rng.uniform(-A + 2.5, A - 2.5, size=2)
            boxes.append(Box(cx, cy, length / 2, 0.08, 0.0 if horiz else np.pi / 2))
        for _ in range(_ri(rng, wc.n_obstacles)):
            cx, cy = rng.uniform(-A + 1.0, A - 1.0, size=2)
            boxes.append(Box(cx, cy, rng.uniform(0.25, 0.75), rng.uniform(0.25, 0.75), rng.uniform(0, np.pi)))

        g = GridMap(A, wc.cell)
        for b in boxes:
            g.add_box(b)
        g.inflate(robot_radius)

        free = np.argwhere(~g.inflated)
        if len(free) < 200:
            continue
        sx, sy = g.to_xy(*free[rng.integers(len(free))])
        reach = g.bfs([g.to_cell(sx, sy)])
        reachable = np.argwhere(reach >= 0)
        if len(reachable) < 200:
            continue

        npe = n_people if n_people is not None else _ri(rng, wc.n_people)
        nfi = n_fires if n_fires is not None else _ri(rng, wc.n_fires)
        targets: List[Tuple[float, float]] = []
        ok = True
        for _ in range(npe + nfi):
            placed = False
            for _ in range(60):
                tx, ty = g.to_xy(*reachable[rng.integers(len(reachable))])
                if np.hypot(tx - sx, ty - sy) < 3.0:
                    continue
                if any(np.hypot(tx - ox, ty - oy) < 2.0 for ox, oy in targets):
                    continue
                targets.append((tx, ty))
                placed = True
                break
            if not placed:
                ok = False
                break
        if not ok:
            continue
        # spawn pose: face roughly towards arena centre, with jitter
        yaw = float(np.arctan2(-sy, -sx) + rng.normal(0, 0.6))
        return WorldLayout(boxes, targets[:npe], targets[npe:], (float(sx), float(sy), yaw), g)
    raise RuntimeError("could not generate a reachable world; relax WorldConfig")
