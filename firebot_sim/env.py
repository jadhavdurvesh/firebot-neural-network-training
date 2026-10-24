"""FireBotEnv: physics-based firefighting/rescue training environment.

Mission (mirrors firebot-core policy): find and reach every person FIRST, then
extinguish every fire, without collisions, overheating or running the battery
flat. All numbers are simulated; nothing here is hardware telemetry.

Action  (3, float in [-1, 1]): [throttle, steering, pump]
Obs     (flat float32): lidar | known-target bearings | body state | pose | coverage
"""
from __future__ import annotations

import math
from typing import Optional

import gymnasium as gym
import numpy as np
import pybullet as pb
import pybullet_data
from gymnasium import spaces
from pybullet_utils import bullet_client

from .config import DOMAIN_RANDOMIZATION as DR
from .config import EnvConfig, RobotConfig, WorldConfig
from . import obs as obsmod
from .robot import FireBotRobot
from .world import generate_world


class FireBotEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(self, robot_cfg: Optional[RobotConfig] = None, world_cfg: Optional[WorldConfig] = None,
                 env_cfg: Optional[EnvConfig] = None, render_mode: Optional[str] = None,
                 n_people=None, n_fires=None):
        super().__init__()
        self.rcfg = robot_cfg or RobotConfig()
        # world_cfg may be a list of WorldConfig: one is drawn per episode (mixed-difficulty training)
        self._wcfgs = list(world_cfg) if isinstance(world_cfg, (list, tuple)) else [world_cfg or WorldConfig()]
        self.wcfg = self._wcfgs[0]
        self.ecfg = env_cfg or EnvConfig()
        self.render_mode = render_mode
        self._n_people, self._n_fires = n_people, n_fires
        self.substeps = self.ecfg.physics_hz // self.ecfg.control_hz
        self.dt = 1.0 / self.ecfg.control_hz
        self.bc = None
        self.robot = None

        self.cov_n = self.ecfg.coverage_grid
        self.obs_dim = self.rcfg.lidar_rays + self.rcfg.ir_bins + 4 + 4 + 9 + 4 + 4 + self.cov_n ** 2 + 3
        self.observation_space = spaces.Box(-np.inf, np.inf, (self.obs_dim,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (3,), np.float32)

    # ------------------------------------------------------------------ setup
    def _connect(self):
        self.bc = bullet_client.BulletClient(connection_mode=pb.DIRECT)

    def _static_box(self, cx, cy, hx, hy, yaw, hz=0.25, rgba=(0.55, 0.55, 0.6, 1)):
        bc = self.bc
        col = bc.createCollisionShape(pb.GEOM_BOX, halfExtents=[hx, hy, hz])
        vis = bc.createVisualShape(pb.GEOM_BOX, halfExtents=[hx, hy, hz], rgbaColor=list(rgba))
        return bc.createMultiBody(0, col, vis, [cx, cy, hz], bc.getQuaternionFromEuler([0, 0, yaw]))

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random
        self.wcfg = self._wcfgs[int(rng.integers(len(self._wcfgs)))]
        ec, rc, wc = self.ecfg, self.rcfg, self.wcfg
        if self.bc is None:
            self._connect()
        bc = self.bc
        bc.resetSimulation()
        bc.setAdditionalSearchPath(pybullet_data.getDataPath())
        bc.setGravity(0, 0, -9.81)
        bc.setTimeStep(1.0 / ec.physics_hz)
        self.plane = bc.loadURDF("plane.urdf")
        bc.changeDynamics(self.plane, -1, lateralFriction=1.0)

        A = wc.arena_half
        self.layout = generate_world(rng, wc, rc.robot_radius, self._n_people, self._n_fires)
        L = self.layout
        bld = wc.layout == "building"
        for cx, cy, hx, hy in ((A + 0.1, 0, 0.1, A + 0.2), (-A - 0.1, 0, 0.1, A + 0.2),
                               (0, A + 0.1, A + 0.2, 0.1), (0, -A - 0.1, A + 0.2, 0.1)):
            self._static_box(cx, cy, hx, hy, 0, hz=0.6 if bld else 0.3,
                             rgba=(0.62, 0.58, 0.52, 1) if bld else (0.3, 0.3, 0.35, 1))
        for b in L.boxes:
            wall = b.h > 0.3
            self._static_box(b.cx, b.cy, b.hx, b.hy, b.yaw, hz=b.h,
                             rgba=(0.86, 0.83, 0.76, 1) if wall else (0.55, 0.55, 0.6, 1))
        self.person_ids, self.fire_ids = [], []
        for (px, py) in L.people:
            col = bc.createCollisionShape(pb.GEOM_CYLINDER, radius=wc.person_radius, height=1.0)
            vis = bc.createVisualShape(pb.GEOM_CYLINDER, radius=wc.person_radius, length=1.0,
                                       rgbaColor=[0.15, 0.35, 0.95, 1])
            self.person_ids.append(bc.createMultiBody(0, col, vis, [px, py, 0.5]))
        for (fx, fy) in L.fires:
            vis = bc.createVisualShape(pb.GEOM_SPHERE, radius=wc.fire_radius * 0.8, rgbaColor=[1.0, 0.45, 0.0, 1])
            self.fire_ids.append(bc.createMultiBody(0, -1, vis, [fx, fy, 0.28]))

        # ---- domain randomisation (episode level)
        if ec.randomize:
            mass_s, fric_s, torq_s = (rng.uniform(*DR["chassis_mass_scale"]), rng.uniform(*DR["wheel_friction_scale"]),
                                      rng.uniform(*DR["motor_torque_scale"]))
            self._delay = int(rng.integers(0, ec.max_action_delay + 1))
            self._lidar_noise = rc.lidar_noise_std * rng.uniform(*DR["lidar_noise_scale"])
            self._pose_noise = (DR["pose_noise_std"]["position_m"], DR["pose_noise_std"]["yaw_rad"])
            batt0 = rng.uniform(*DR["initial_battery_fraction"])
            self.smoke = float(rng.uniform(*DR["smoke_level"]))
            T_amb = rc.thermal_ambient_c + rng.uniform(*DR["ambient_temp_offset_c"])
        else:
            mass_s = fric_s = torq_s = 1.0
            self._delay, self._lidar_noise, self._pose_noise = 0, rc.lidar_noise_std, (0.0, 0.0)
            batt0, T_amb = 1.0, rc.thermal_ambient_c
            self.smoke = 0.0
        sx, sy, syaw = L.spawn
        self.robot = FireBotRobot(bc, rc, (sx, sy), syaw, mass_s, fric_s, torq_s)
        for _ in range(30):                      # settle on wheels
            self.robot.physics_tick()
            bc.stepSimulation()

        # ---- mission / internal state
        self.n_people, self.n_fires = len(L.people), len(L.fires)
        self.known_p = np.zeros(self.n_people, bool)
        self.located = np.zeros(self.n_people, bool)
        self.known_f = np.zeros(self.n_fires, bool)
        self.fire_int = np.ones(self.n_fires, np.float32)
        self.vis_p = np.zeros(self.n_people, bool)
        self.vis_f = np.zeros(self.n_fires, bool)
        self.energy_wh = batt0 * rc.battery_wh
        self.water = 1.0
        self.T_amb = T_amb
        self.temp = T_amb
        self.cooling_eff = 1.0
        self.fault_step = (int(rng.integers(150, ec.max_steps // 2))
                           if rng.random() < wc.cooling_failure_prob else None)
        self.visited = np.zeros((self.cov_n, self.cov_n), bool)
        self._act_buf = [np.zeros(3, np.float32) for _ in range(self._delay)]
        self._last_a = np.zeros(3, np.float32)
        self._terms, self._ep_terms = {}, {}
        self.steps = 0
        self.collisions = 0
        self.max_temp = self.temp
        self._goal_key, self._goal_dist = None, None
        self._lidar = np.ones(rc.lidar_rays, np.float32)
        self._ir = np.zeros(rc.ir_bins, np.float32)
        self._sense(first=True)
        self._update_coverage()
        return self._obs(), self._info()

    # ----------------------------------------------------------------- sensing
    def _cam_pos(self, pos, yaw):
        mx, my, mz = self.rcfg.camera_mount
        c, s = math.cos(yaw), math.sin(yaw)
        return np.array([pos[0] + c * mx - s * my, pos[1] + s * mx + c * my, mz])

    def _sense(self, first=False):
        rc, bc, rng = self.rcfg, self.bc, self.np_random
        pos, yaw, _ = self.robot.pose()
        # smoke (randomised, fades as fires go out) degrades lidar and camera
        smoke = self.smoke * (float(self.fire_int.mean()) if self.n_fires else 0.0)
        det_range = rc.detection_range * (1.0 - 0.5 * smoke)
        # --- lidar
        n = rc.lidar_rays
        ang = yaw + 2 * np.pi * np.arange(n) / n
        o = [pos[0], pos[1], 0.30]
        frm = [o] * n
        to = [[o[0] + rc.lidar_range * math.cos(a), o[1] + rc.lidar_range * math.sin(a), 0.30] for a in ang]
        res = bc.rayTestBatch(frm, to)
        fr = np.array([r[2] if r[0] not in (-1, self.robot.id) else 1.0 for r in res], np.float32)
        rng_m = fr * rc.lidar_range + rng.normal(0, self._lidar_noise * (1 + smoke), n)
        drop = rng.random(n) < (rc.lidar_dropout + 0.08 * smoke)
        fr = np.clip(rng_m / rc.lidar_range, 0, 1).astype(np.float32)
        fr[drop] = 1.0
        self._lidar = fr
        # --- camera "detector" (simulated vision model; replaced by firebot-vision later)
        cam = self._cam_pos(pos, yaw)
        half_fov = math.radians(rc.camera_fov_deg) / 2

        def check(points, z_end, ids, is_person):
            vis = np.zeros(len(points), bool)
            cand, frm_l, to_l = [], [], []
            for i, (tx, ty) in enumerate(points):
                dx, dy = tx - cam[0], ty - cam[1]
                d = math.hypot(dx, dy)
                b = math.atan2(dy, dx) - yaw
                b = (b + math.pi) % (2 * math.pi) - math.pi
                if d <= det_range and abs(b) <= half_fov:
                    cand.append(i)
                    frm_l.append(cam.tolist())
                    to_l.append([tx, ty, z_end])
            if cand:
                for i, r in zip(cand, bc.rayTestBatch(frm_l, to_l)):
                    hit = r[0]
                    if is_person:
                        vis[i] = hit == ids[i]
                    else:
                        vis[i] = hit in (-1, self.robot.id)
            return vis

        self.vis_p = check(self.layout.people, 0.5, self.person_ids, True)
        self.vis_f = check(self.layout.fires, 0.28, self.fire_ids, False) & (self.fire_int > 0)

        # --- forward IR strip: heat from active fires per bearing bin. Walls block it; smoke does NOT.
        nb, fov = rc.ir_bins, math.radians(rc.ir_fov_deg)
        centers = -fov / 2 + (np.arange(nb) + 0.5) * fov / nb
        ir = np.zeros(nb, np.float32)
        cand, frm_l, to_l = [], [], []
        for i, (fx, fy) in enumerate(self.layout.fires):
            if self.fire_int[i] <= 0:
                continue
            dx, dy = fx - cam[0], fy - cam[1]
            d = math.hypot(dx, dy)
            b = (math.atan2(dy, dx) - yaw + math.pi) % (2 * math.pi) - math.pi
            if d <= rc.ir_range and abs(b) <= fov / 2 + fov / nb:
                cand.append((i, d, b))
                frm_l.append(cam.tolist())
                to_l.append([fx, fy, 0.28])
        if cand:
            for (i, d, b), r in zip(cand, bc.rayTestBatch(frm_l, to_l)):
                if r[0] in (-1, self.robot.id):                     # clear line to the fire
                    amp = float(self.fire_int[i]) / (1.0 + (d / 2.5) ** 2)
                    ir += (amp * np.exp(-0.5 * ((centers - b) / (fov / nb)) ** 2)).astype(np.float32)
        ir += rng.normal(0, rc.ir_noise_std, nb).astype(np.float32)
        self._ir = np.clip(ir, 0.0, None)
        return pos, yaw

    def _update_coverage(self):
        pos, _, _ = self.robot.pose()
        return obsmod.update_coverage(self.visited, pos[0], pos[1], self.wcfg.arena_half)

    # ------------------------------------------------------------- observation
    def _obs(self):
        rc = self.rcfg
        A = self.wcfg.arena_half
        pos, yaw, _ = self.robot.pose()
        vx, vy, wz = self.robot.body_velocity()
        pn, yn = self._pose_noise
        rng = self.np_random
        x, y = pos[0] + rng.normal(0, pn), pos[1] + rng.normal(0, pn)
        yw = yaw + rng.normal(0, yn)
        p_feat = obsmod.target_feat(x, y, yw, self.layout.people, self.known_p & ~self.located, A)
        f_feat = obsmod.target_feat(x, y, yw, self.layout.fires, self.known_f & (self.fire_int > 0), A)
        body = obsmod.body_vec(vx, vy, wz, self.robot.steer_actual, rc.max_speed, rc.max_steer,
                               self.energy_wh / rc.battery_wh, self.water, self.temp, self.T_amb,
                               rc.thermal_critical_c, self.cooling_eff, self.located.all())
        mission = obsmod.mission_vec(int(self.located.sum()), self.n_people,
                                     self.fire_int.sum() > 0, self.n_fires)
        return obsmod.assemble(self._lidar, p_feat, f_feat, body, mission,
                               obsmod.pose_vec(x, y, yw, A), self.visited, self._last_a, self._ir)

    def _info(self, reason=None):
        return {"people_located": int(self.located.sum()), "n_people": self.n_people,
                "fires_out": int((self.fire_int <= 0).sum()), "n_fires": self.n_fires,
                "collisions": self.collisions, "battery": max(0.0, self.energy_wh) / self.rcfg.battery_wh,
                "temp_c": self.temp, "max_temp_c": self.max_temp, "water": self.water,
                "steps": self.steps, "end_reason": reason, "simulated": True,
                "reward_terms": dict(self._ep_terms)}

    # -------------------------------------------------------------------- step
    def _goal(self):
        """Nearest KNOWN goal under the person-first policy, or None (=> explore)."""
        pos, _, _ = self.robot.pose()
        cand = [(math.hypot(px - pos[0], py - pos[1]), ("p", i))
                for i, (px, py) in enumerate(self.layout.people) if self.known_p[i] and not self.located[i]]
        if not cand and self.located.all():
            cand = [(math.hypot(fx - pos[0], fy - pos[1]), ("f", i))
                    for i, (fx, fy) in enumerate(self.layout.fires) if self.known_f[i] and self.fire_int[i] > 0]
        return min(cand) if cand else (None, None)

    def _rw(self, name: str, value: float) -> float:
        """Record one named reward term ('points') for this step and return it."""
        value = float(value)
        self._terms[name] = self._terms.get(name, 0.0) + value
        self._ep_terms[name] = self._ep_terms.get(name, 0.0) + value
        return value

    def step(self, action):
        ec, rc = self.ecfg, self.rcfg
        a = np.clip(np.asarray(action, np.float32), -1, 1)
        self._last_a = a.copy()
        self._act_buf.append(a)
        a = self._act_buf.pop(0)
        thr, st, pump = float(a[0]), float(a[1]), float(a[2])

        r = self.robot
        r.set_command(thr * rc.max_speed, st * rc.max_steer)
        r.update_steering(self.dt)
        pump_on = pump > 0 and self.water > 0
        p_motor = 0.0
        for _ in range(self.substeps):
            p_motor += r.physics_tick()
            self.bc.stepSimulation()
        p_motor /= self.substeps
        self.steps += 1
        self._terms = {}
        reward = self._rw("time_penalty", ec.r_time)

        # ---- collisions (anything but the ground)
        cps = self.bc.getContactPoints(bodyA=r.id)
        if any(cp[2] != self.plane for cp in cps):
            self.collisions += 1
            reward += self._rw("collision", ec.r_collision)

        # ---- sensing and knowledge
        pos, yaw = self._sense()
        rng = self.np_random
        for i in range(self.n_people):
            if self.vis_p[i] and not self.known_p[i] and rng.random() < rc.detection_prob:
                self.known_p[i] = True
                reward += self._rw("detect_person", ec.r_detect)
        for i in range(self.n_fires):
            if self.vis_f[i] and not self.known_f[i] and rng.random() < rc.detection_prob:
                self.known_f[i] = True
                reward += self._rw("detect_fire", ec.r_detect * 0.5)
        for i, (px, py) in enumerate(self.layout.people):
            d = math.hypot(px - pos[0], py - pos[1])
            if not self.located[i] and self.known_p[i] and d <= ec.locate_dist and (self.vis_p[i] or d < 1.0):
                self.located[i] = True
                reward += self._rw("person_located", ec.r_locate)
        people_done = bool(self.located.all())

        # ---- pump / fire suppression (only counts once all people are located)
        hit_any = False
        if pump_on:
            self.water = max(0.0, self.water - rc.pump_water_rate * self.dt)
            for i, (fx, fy) in enumerate(self.layout.fires):
                if self.fire_int[i] <= 0 or not self.known_f[i]:
                    continue
                dx, dy = fx - pos[0], fy - pos[1]
                d = math.hypot(dx, dy)
                b = (math.atan2(dy, dx) - yaw + math.pi) % (2 * math.pi) - math.pi
                if people_done and d <= rc.pump_range and abs(b) <= rc.pump_half_angle and self.vis_f[i]:
                    hit_any = True
                    self.fire_int[i] -= rc.pump_suppress_rate * self.dt
                    if self.fire_int[i] <= 0:
                        self.fire_int[i] = 0.0
                        reward += self._rw("fire_extinguished", ec.r_fire_out)
                        self.bc.changeVisualShape(self.fire_ids[i], -1, rgbaColor=[0.2, 0.2, 0.2, 1])
            if not hit_any:
                reward += self._rw("wasted_pump", ec.r_wasted_pump)

        # ---- power and thermal (synthetic models)
        p_tot = rc.idle_power_w + p_motor + (rc.pump_power_w if pump_on else 0.0)
        self.energy_wh -= p_tot * self.dt * rc.energy_time_scale / 3600.0
        if self.fault_step is not None and self.steps >= self.fault_step:
            self.cooling_eff = 0.15
        fire_term = 0.0
        for i, (fx, fy) in enumerate(self.layout.fires):
            if self.fire_int[i] > 0:
                d = math.hypot(fx - pos[0], fy - pos[1])
                fire_term += rc.fire_heat_rate * self.fire_int[i] * max(0.0, 1 - d / rc.fire_heat_radius) ** 2
        self.temp += self.dt * (p_tot / rc.thermal_capacity + fire_term
                                - rc.thermal_cooling_k * self.cooling_eff * (self.temp - self.T_amb))
        self.max_temp = max(self.max_temp, self.temp)
        span = rc.thermal_critical_c - rc.thermal_warn_c
        reward += self._rw("thermal_risk", ec.r_thermal * max(0.0, (self.temp - rc.thermal_warn_c) / span))

        # ---- exploration + goal-progress shaping
        reward += self._rw("exploration", ec.r_explore * self._update_coverage())
        d, key = self._goal()
        if key is not None and key == self._goal_key and self._goal_dist is not None:
            reward += self._rw("progress_to_goal", ec.r_progress * (self._goal_dist - d))
        self._goal_key, self._goal_dist = key, d

        # ---- termination
        terminated, reason = False, None
        _, _, up_z = r.pose()
        if self.located.all() and (self.fire_int <= 0).all():
            terminated, reason = True, "success"
            reward += self._rw("mission_success", ec.r_success)
        elif self.temp >= rc.thermal_critical_c:
            terminated, reason = True, "thermal_critical"
            reward += self._rw("fail_thermal", ec.r_fail)
        elif self.energy_wh <= 0:
            terminated, reason = True, "battery_empty"
            reward += self._rw("fail_battery", ec.r_fail)
        elif up_z < 0.3:
            terminated, reason = True, "rollover"
            reward += self._rw("fail_rollover", ec.r_fail)
        truncated = (not terminated) and self.steps >= ec.max_steps
        if truncated:
            reason = "timeout"
        return self._obs(), float(reward), terminated, truncated, self._info(reason)

    # ----------------------------------------------------------- export / render
    def firebot_messages(self) -> dict:
        """Current state as INTERFACES.md envelopes (status always 'simulated')."""
        import datetime
        pos, yaw, _ = self.robot.pose()
        ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
        env = {"schema_version": "0.1", "source": "simulation-physics", "robot_id": "FB-01",
               "timestamp": ts, "status": "simulated"}
        vx, vy, wz = self.robot.body_velocity()
        return {
            "robot_state": {**env, "payload": {
                "mode": "autonomous", "pose": {"x": float(pos[0]), "y": float(pos[1]), "yaw": float(yaw)},
                "health": {"temp_c": self.temp, "cooling_effectiveness": self.cooling_eff},
                "battery_state": self.energy_wh / self.rcfg.battery_wh,
                "pump_water_state": self.water, "speed_mps": float(vx)}},
            "detections": [{**env, "source": "vision", "payload": {
                "detection_id": f"person-{i}", "class": "person", "confidence": 0.9,
                "location": list(self.layout.people[i]), "priority": "high"}}
                for i in range(self.n_people) if self.known_p[i]] +
                [{**env, "source": "vision", "payload": {
                    "detection_id": f"fire-{i}", "class": "fire", "confidence": 0.9,
                    "location": list(self.layout.fires[i]), "priority": "normal"}}
                 for i in range(self.n_fires) if self.known_f[i] and self.fire_int[i] > 0],
        }

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        A = self.wcfg.arena_half
        view = self.bc.computeViewMatrix([0, 0, 2.2 * A], [0, 0, 0], [0, 1, 0])
        proj = self.bc.computeProjectionMatrixFOV(45, 1.0, 0.1, 50)
        _, _, rgb, _, _ = self.bc.getCameraImage(480, 480, view, proj, renderer=pb.ER_TINY_RENDERER)
        return np.reshape(rgb, (480, 480, 4))[:, :, :3].astype(np.uint8)

    def close(self):
        if self.bc is not None:
            self.bc.disconnect()
            self.bc = None
