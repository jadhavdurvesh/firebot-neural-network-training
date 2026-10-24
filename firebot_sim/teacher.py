"""Reactive teacher for imitation learning (DAgger) -- uses ONLY information the student also observes
(known targets, coverage memory, noisy lidar, pose), so the student can actually learn to imitate it.

 * known goal (people first, then fires): drive towards it with lidar obstacle avoidance
 * no known goal: head for the nearest unexplored coverage cell (exploration)
 * stuck recovery: a car cannot turn on the spot, so it backs out
"""
from __future__ import annotations

import math

import numpy as np


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class ReactiveTeacher:
    def __init__(self, env):
        self.env = env
        self._stuck = 0
        self._rev = 0
        self._rev_steer = 0.0
        self._cell_goal = None

    def goal(self):
        e = self.env
        pos, _, _ = e.robot.pose()
        ppl = [(math.hypot(x - pos[0], y - pos[1]), (x, y), "p")
               for i, (x, y) in enumerate(e.layout.people) if e.known_p[i] and not e.located[i]]
        if ppl:
            return min(ppl)
        if e.located.all():
            fires = [(math.hypot(x - pos[0], y - pos[1]), (x, y), "f")
                     for i, (x, y) in enumerate(e.layout.fires) if e.known_f[i] and e.fire_int[i] > 0]
            if fires:
                return min(fires)
        return None

    def _explore_target(self, pos):
        e = self.env
        A, N = e.wcfg.arena_half, e.cov_n
        cs = 2 * A / N
        if self._cell_goal is not None and not e.visited[self._cell_goal]:
            ix, iy = self._cell_goal
        else:
            cand = [(ix, iy) for ix in range(N) for iy in range(N) if not e.visited[ix, iy]]
            if not cand:
                ix, iy = int(e.np_random.integers(N)), int(e.np_random.integers(N))
                self._cell_goal = None
            else:
                ix, iy = min(cand, key=lambda c: math.hypot(-A + (c[0] + .5) * cs - pos[0],
                                                            -A + (c[1] + .5) * cs - pos[1]))
                self._cell_goal = (ix, iy)
        return -A + (ix + 0.5) * cs, -A + (iy + 0.5) * cs

    def act(self):
        e, rc = self.env, self.env.rcfg
        pos, yaw, _ = e.robot.pose()
        vx, _, _ = e.robot.body_velocity()
        lidar = e._lidar * rc.lidar_range
        n = len(lidar)
        ang = np.array([_wrap(2 * math.pi * k / n) for k in range(n)])
        front = float(lidar[np.abs(ang) < 0.5].min())
        left = float(lidar[(ang > 0.3) & (ang < 1.4)].mean())
        right = float(lidar[(ang < -0.3) & (ang > -1.4)].mean())

        g = self.goal()
        pump = -1.0
        ir = e._ir
        if g is None and e.located.all() and ir.max() > 0.05:
            # people are done and the IR strip sees heat: turn towards the hottest bearing
            fov = math.radians(rc.ir_fov_deg)
            err = -fov / 2 + (int(np.argmax(ir)) + 0.5) * fov / len(ir)
            d, kind = 5.0, "x"
        else:
            if g is not None:
                d, (tx, ty), kind = g
            else:
                tx, ty = self._explore_target(pos)
                d, kind = math.hypot(tx - pos[0], ty - pos[1]), "x"
            err = _wrap(math.atan2(ty - pos[1], tx - pos[0]) - yaw)

        if kind == "f" and d <= 2.2:                      # in pump range: stop, aim, spray
            if abs(err) < 0.4:
                pump = 1.0
            return np.array([0.0, float(np.clip(1.8 * err, -1, 1)), pump], np.float32)

        # --- smooth avoidance (continuous in lidar) + counter-based back-out (identifiable via last action in obs)
        steer = float(np.clip(1.8 * err, -1, 1))
        side = float(np.clip((left - right) / 2.0, -1.0, 1.0))      # >0: more room on the left
        if front < 1.4:
            w = float(np.clip((1.4 - front) / 1.0, 0, 1))
            steer = (1 - w) * steer + w * side
        speed = 0.7 * max(0.25, math.cos(err)) if abs(err) < 1.2 else 0.3
        speed *= float(np.clip((front - 0.4) / 1.0, 0.2, 1.0))

        self._stuck = self._stuck + 1 if abs(vx) < 0.05 else 0
        if self._stuck > 20 and self._rev == 0:
            self._rev, self._stuck = 25, 0
            self._rev_steer = -1.0 if left > right else 1.0
        if self._rev > 0:
            self._rev -= 1
            return np.array([-0.6, self._rev_steer, -1.0], np.float32)
        return np.array([speed, steer, pump], np.float32)
