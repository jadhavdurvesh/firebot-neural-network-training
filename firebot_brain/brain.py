"""FireBotBrain: sensor frame in -> safe motor command out. Runs a trained ONNX policy.

    brain = FireBotBrain("firebot_policy.onnx", n_people=2, n_fires=1)
    cmd, info = brain.step(frame)          # call at 20 Hz (the control rate used in training)
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np

from firebot_sim import obs as obsmod
from firebot_sim.config import EnvConfig, RobotConfig, WorldConfig
from .hal import Command, SensorFrame
from .memory import MissionMemory
from .safety import SafetySupervisor


class FireBotBrain:
    def __init__(self, onnx_path: Optional[str], robot_cfg: RobotConfig | None = None,
                 n_people: int = 1, n_fires: int = 1, arena_half: float | None = None,
                 safety: bool = True, env_cfg: EnvConfig | None = None):
        self.rc = robot_cfg or RobotConfig()
        self.ec = env_cfg or EnvConfig()
        self.A = arena_half if arena_half is not None else WorldConfig().arena_half
        self.mem = MissionMemory(n_people, n_fires, self.A, self.ec.locate_dist, self.ec.coverage_grid)
        self.n_people, self.n_fires = n_people, n_fires
        self.safety = SafetySupervisor(self.rc) if safety else None
        self._last = np.zeros(3, np.float32)       # policy's previous raw output (part of the observation)
        self.sess = None
        if onnx_path:
            import onnxruntime as ort
            self.sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
            self.expected_dim = self.sess.get_inputs()[0].shape[1]

    def reset(self):
        self.mem.reset()
        self._last = np.zeros(3, np.float32)

    def build_obs(self, f: SensorFrame) -> np.ndarray:
        """Exactly the simulator's observation layout (see firebot_sim/obs.py)."""
        rc, m = self.rc, self.mem
        lidar = np.clip(np.asarray(f.lidar_m, np.float32) / rc.lidar_range, 0.0, 1.0)
        known_unlocated = [not p["located"] for p in m.people]
        pf = obsmod.target_feat(f.x, f.y, f.yaw, [(p["x"], p["y"]) for p in m.people], known_unlocated, self.A)
        ff = obsmod.target_feat(f.x, f.y, f.yaw, [(q["x"], q["y"]) for q in m.fires],
                                [q["active"] for q in m.fires], self.A)
        body = obsmod.body_vec(f.vx, f.vy, f.wz, f.steer_angle, rc.max_speed, rc.max_steer, f.battery_frac,
                               f.water_frac, f.temp_c, f.ambient_c, rc.thermal_critical_c, f.cooling_eff,
                               m.all_people_located)
        mission = obsmod.mission_vec(m.located_count, self.n_people, m.any_fire_active, self.n_fires)
        return obsmod.assemble(lidar, pf, ff, body, mission, obsmod.pose_vec(f.x, f.y, f.yaw, self.A), m.visited, self._last, f.ir_bins)

    def step(self, f: SensorFrame, now: float | None = None):
        t0 = time.perf_counter()
        self.mem.update(f)
        obs = self.build_obs(f)
        if self.sess is None:
            raise RuntimeError("no ONNX policy loaded")
        if obs.shape[0] != self.expected_dim:
            raise ValueError(f"obs dim {obs.shape[0]} != policy input {self.expected_dim}: robot config / lidar rays differ from training")
        a = self.sess.run(None, {"obs": obs[None]})[0][0]
        raw = Command(float(a[0]), float(a[1]), float(a[2]))
        self._last = np.clip(a, -1, 1).astype(np.float32)
        cmd, why = (self.safety.filter(f, raw, now) if self.safety else (raw, []))
        return cmd, {"raw": raw, "safety": why, "latency_ms": (time.perf_counter() - t0) * 1e3,
                     "people_located": self.mem.located_count, "known_fires": len(self.mem.fires)}
