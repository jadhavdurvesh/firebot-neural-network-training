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
    h: float = 0.25          # half-height (walls are taller than furniture)


@dataclass
class WorldLayout:
    boxes: List[Box]
    people: List[Tuple[float, float]]
    fires: List[Tuple[float, float]]
    spawn: Tuple[float, float, float]            # x, y, yaw
    grid: "GridMap" = None
    rooms: List = field(default_factory=list)   # building layout: (x0, x1, y0, y1) per room


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
    if wc.layout == "building":
        return generate_building(rng, wc, robot_radius, n_people, n_fires)
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


def _split(rng, lo, hi, k, min_w):
    """Split [lo, hi] into k intervals, each at least min_w wide."""
    k = max(1, min(k, int((hi - lo) // min_w)))
    spare = (hi - lo) - k * min_w
    w = min_w + rng.dirichlet(np.ones(k)) * spare
    return list(lo + np.concatenate([[0.0], np.cumsum(w)]))


def generate_building(rng: np.random.Generator, wc: WorldConfig, robot_radius: float,
                      n_people=None, n_fires=None, max_tries: int = 300) -> WorldLayout:
    """Rooms on both sides of a hallway. Every room has a door onto the hallway; some rooms also connect to
    their neighbours. People and fires are placed inside rooms; the robot starts in the hallway."""
    A, T, HW, DW = wc.arena_half, 0.075, 0.6, wc.door_width
    for _ in range(max_tries):
        hy = rng.uniform(-1.0, 1.0)
        y_lo, y_hi = hy - wc.hall_width / 2, hy + wc.hall_width / 2
        boxes: List[Box] = []
        rooms, door_c = [], []
        for side in (+1, -1):
            wall_y = y_hi if side > 0 else y_lo
            far_y = A if side > 0 else -A
            ya, yb = sorted((wall_y, far_y))
            cuts = _split(rng, -A, A, _ri(rng, wc.n_rooms_per_side), 2.8)
            doors = []
            for i in range(len(cuts) - 1):
                x0, x1 = cuts[i], cuts[i + 1]
                lo, hi = x0 + DW / 2 + 0.5, x1 - DW / 2 - 0.5
                dx = rng.uniform(lo, hi) if hi > lo else (x0 + x1) / 2
                doors.append((dx - DW / 2, dx + DW / 2))
                rooms.append((x0, x1, ya, yb))
                door_c.append((dx, wall_y))
            edges = [-A] + [v for d in doors for v in d] + [A]
            for j in range(0, len(edges), 2):                   # hallway-side wall between the doors
                a, b = edges[j], edges[j + 1]
                if b - a > 0.05:
                    boxes.append(Box((a + b) / 2, wall_y, (b - a) / 2, T, 0.0, HW))
            for i in range(1, len(cuts) - 1):                   # dividers between neighbouring rooms
                x = cuts[i]
                if rng.random() < 0.4 and (yb - ya) > 3.0:      # connecting door
                    dyc = rng.uniform(ya + DW / 2 + 0.5, yb - DW / 2 - 0.5)
                    segs = [(ya, dyc - DW / 2), (dyc + DW / 2, yb)]
                else:
                    segs = [(ya, yb)]
                for a, b in segs:
                    if b - a > 0.05:
                        boxes.append(Box(x, (a + b) / 2, T, (b - a) / 2, 0.0, HW))
        for (x0, x1, ya, yb), (dcx, dcy) in zip(rooms, door_c):    # furniture, kept clear of the main door
            for _ in range(_ri(rng, wc.room_furniture)):
                hx, hy_ = rng.uniform(0.25, 0.5, size=2)
                if x1 - x0 < 2 * (hx + 0.9) or yb - ya < 2 * (hy_ + 0.9):
                    continue
                cx = rng.uniform(x0 + hx + 0.9, x1 - hx - 0.9)
                cy = rng.uniform(ya + hy_ + 0.9, yb - hy_ - 0.9)
                if abs(cx - dcx) < 1.3 and abs(cy - dcy) < 1.8:
                    continue
                boxes.append(Box(cx, cy, hx, hy_, rng.uniform(0, np.pi)))

        g = GridMap(A, wc.cell)
        for b in boxes:
            g.add_box(b)
        g.inflate(robot_radius)
        hall = [(ix, iy) for ix, iy in np.argwhere(~g.inflated) if y_lo + 0.1 < g.to_xy(ix, iy)[1] < y_hi - 0.1]
        if len(hall) < 50:
            continue
        sx, sy = g.to_xy(*hall[rng.integers(len(hall))])
        reach = g.bfs([g.to_cell(sx, sy)])
        room_cells = []
        for (x0, x1, ya, yb) in rooms:
            cells = [(ix, iy) for ix, iy in np.argwhere(reach >= 0)
                     if x0 + 0.5 < g.to_xy(ix, iy)[0] < x1 - 0.5 and ya + 0.5 < g.to_xy(ix, iy)[1] < yb - 0.5]
            room_cells.append(cells)
        usable = [i for i, c in enumerate(room_cells) if len(c) > 15]
        npe = n_people if n_people is not None else _ri(rng, wc.n_people)
        nfi = n_fires if n_fires is not None else _ri(rng, wc.n_fires)
        if len(usable) < 2:
            continue
        order = list(rng.permutation(usable))
        targets: List[Tuple[float, float]] = []
        ok = True
        for n in range(npe + nfi):
            placed = False
            cells = room_cells[order[n % len(order)]]
            for _ in range(80):
                tx, ty = g.to_xy(*cells[rng.integers(len(cells))])
                if np.hypot(tx - sx, ty - sy) < 3.0 or any(np.hypot(tx - ox, ty - oy) < 2.0 for ox, oy in targets):
                    continue
                targets.append((tx, ty))
                placed = True
                break
            if not placed:
                ok = False
                break
        if not ok:
            continue
        yaw = float(rng.choice([0.0, np.pi]) + rng.normal(0, 0.3))
        return WorldLayout(boxes, targets[:npe], targets[npe:], (float(sx), float(sy), yaw), g, rooms)
    raise RuntimeError("could not generate a building; relax WorldConfig")
