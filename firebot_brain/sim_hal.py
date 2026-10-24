"""SimHAL: lets the exact on-robot brain code run against the physics simulator (sim-in-the-loop test)."""
from __future__ import annotations

import numpy as np

from .hal import Command, Detection, SensorFrame


class SimHAL:
    def __init__(self, env):
        self.env = env
        self._fresh = True

    def reset(self, seed=None):
        obs, info = self.env.reset(seed=seed)
        self._fresh = True
        return obs, info

    def read(self) -> SensorFrame:
        e, rc = self.env, self.env.rcfg
        pos, yaw, _ = e.robot.pose()
        vx, vy, wz = e.robot.body_velocity()
        dets = []
        if not self._fresh:                      # no detections before the first step (matches sim)
            for i, (x, y) in enumerate(e.layout.people):
                if e.vis_p[i]:
                    dets.append(Detection("person", x, y))
            for i, (x, y) in enumerate(e.layout.fires):
                if e.known_f[i] and e.fire_int[i] <= 0:
                    dets.append(Detection("fire", x, y, active=False))
                elif e.vis_f[i]:
                    dets.append(Detection("fire", x, y, active=True))
        return SensorFrame(
            t=e.steps * e.dt, x=float(pos[0]), y=float(pos[1]), yaw=float(yaw), vx=vx, vy=vy, wz=wz,
            steer_angle=e.robot.steer_actual, lidar_m=e._lidar * rc.lidar_range, detections=dets, ir_bins=e._ir.copy(),
            battery_frac=max(0.0, e.energy_wh) / rc.battery_wh, water_frac=e.water, temp_c=e.temp,
            ambient_c=e.T_amb, cooling_eff=e.cooling_eff)

    def send(self, cmd: Command):
        self._fresh = False
        return self.env.step(np.array([cmd.throttle, cmd.steer, cmd.pump], np.float32))
