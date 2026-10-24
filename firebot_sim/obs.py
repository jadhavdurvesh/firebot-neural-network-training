"""Observation construction shared by the simulator AND the on-robot brain.

Keeping this in one place is what prevents train/deploy skew: the robot builds
its observation vector with exactly the same code the policy was trained with.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def target_feat(x: float, y: float, yaw: float, pts: Sequence, mask: Sequence, arena_half: float):
    """[known, sin(bearing), cos(bearing), dist/diag] of the nearest masked target, robot frame."""
    best = None
    c, s = math.cos(yaw), math.sin(yaw)
    for (px, py), m in zip(pts, mask):
        if not m:
            continue
        dx, dy = px - x, py - y
        d = math.hypot(dx, dy)
        if best is None or d < best[0]:
            best = (d, c * dx + s * dy, -s * dx + c * dy)
    if best is None:
        return [0.0, 0.0, 0.0, 0.0]
    d, bx, by = best
    b = math.atan2(by, bx)
    return [1.0, math.sin(b), math.cos(b), d / (2 * arena_half * 1.414)]


def update_coverage(visited: np.ndarray, x: float, y: float, arena_half: float, radius: float = 1.5) -> int:
    """Mark coarse arena cells near the robot as explored (in place). Returns number of new cells."""
    n = visited.shape[0]
    cs = 2 * arena_half / n
    centers = -arena_half + (np.arange(n) + 0.5) * cs
    d = np.hypot(centers[:, None] - x, centers[None, :] - y)
    near = (d <= radius + cs * 0.3) & ~visited
    visited |= near
    return int(near.sum())


def body_vec(vx, vy, wz, steer_actual, max_speed, max_steer, battery_frac, water_frac,
             temp_c, ambient_c, critical_c, cooling_eff, all_people_located):
    span = critical_c - ambient_c
    return [vx / max_speed, vy / max_speed, wz / 2.0, steer_actual / max_steer,
            max(0.0, battery_frac), water_frac, (temp_c - ambient_c) / span, cooling_eff,
            float(all_people_located)]


def mission_vec(located_count, n_people, any_fire_active, n_fires):
    return [1.0 - located_count / max(1, n_people), float(any_fire_active), n_people / 4.0, n_fires / 4.0]


def pose_vec(x, y, yaw, arena_half):
    return [x / arena_half, y / arena_half, math.cos(yaw), math.sin(yaw)]


def assemble(lidar_norm, p_feat, f_feat, body, mission, pose, visited, last_action, ir) -> np.ndarray:
    """last_action = the policy's previous [throttle, steer, pump] output (gives the policy short-term memory)."""
    return np.concatenate([np.asarray(lidar_norm, np.float32), np.asarray(ir, np.float32), p_feat, f_feat, body, mission, pose,
                           visited.flatten().astype(np.float32),
                           np.asarray(last_action, np.float32)]).astype(np.float32)
