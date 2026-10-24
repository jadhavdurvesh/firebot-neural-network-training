"""Safety supervisor: a deterministic, non-learned layer between the neural policy and the motors.

NOT a certified safety function. Real firefighting use needs independent hardware e-stop,
current/thermal cut-offs and qualified review (see firebot-core HANDOFF.md). This layer exists so
that a bad policy output, stale sensors or a hot/flat system degrades to a stop instead of a crash.
"""
from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np

from firebot_sim.config import RobotConfig
from .hal import Command, SensorFrame


class SafetySupervisor:
    def __init__(self, cfg: RobotConfig | None = None, stop_dist: float = 0.35, slow_dist: float = 1.2,
                 cone_deg: float = 30.0, max_frame_age_s: float = 0.25):
        self.cfg = cfg or RobotConfig()
        self.stop_dist, self.slow_dist = stop_dist, slow_dist
        self.cone = math.radians(cone_deg)
        self.max_age = max_frame_age_s

    def _sector_min(self, lidar_m: np.ndarray, center: float) -> float:
        n = len(lidar_m)
        ang = 2 * math.pi * np.arange(n) / n
        diff = np.abs((ang - center + math.pi) % (2 * math.pi) - math.pi)
        sel = lidar_m[diff <= self.cone]
        return float(sel.min()) if sel.size else float(self.cfg.lidar_range)

    def filter(self, f: SensorFrame, cmd: Command, now: float | None = None) -> Tuple[Command, List[str]]:
        c, why = self.cfg, []
        thr, steer, pump = float(np.clip(cmd.throttle, -1, 1)), float(np.clip(cmd.steer, -1, 1)), cmd.pump
        stop = Command(0.0, 0.0, -1.0)
        if f.estop:
            return stop, ["estop"]
        if f.manual_override:
            return stop, ["manual_override"]          # brain output suppressed; operator drives
        if now is not None and (now - f.t) > self.max_age:
            return stop, ["stale_sensors"]
        if not np.isfinite([thr, steer, float(pump)]).all() or not np.isfinite(f.lidar_m).all():
            return stop, ["non_finite"]
        if f.temp_c >= c.thermal_critical_c:
            return stop, ["thermal_critical"]
        if f.battery_frac <= 0.05:
            return stop, ["battery_critical"]
        if f.temp_c >= c.thermal_warn_c:
            thr = float(np.clip(thr, -0.5, 0.5))
            why.append("thermal_warn")
        if f.battery_frac <= 0.15:
            thr = float(np.clip(thr, -0.6, 0.6))
            why.append("battery_low")
        # obstacle clearance in the direction of travel
        if thr > 0:
            d = self._sector_min(f.lidar_m, 0.0)
        elif thr < 0:
            d = self._sector_min(f.lidar_m, math.pi)
        else:
            d = None
        if d is not None and d < self.slow_dist:
            allowed = float(np.clip((d - self.stop_dist) / (self.slow_dist - self.stop_dist), 0.0, 1.0))
            if abs(thr) > allowed:
                thr = math.copysign(allowed, thr)
                why.append("obstacle_limit")
        if f.water_frac <= 0 and pump > 0:
            pump = -1.0
            why.append("pump_no_water")
        return Command(thr, steer, pump), why
