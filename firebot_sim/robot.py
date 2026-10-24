"""Car-like (Ackermann) robot for PyBullet, generated from RobotConfig.

The motor is a simple DC torque-speed model (linear drop from stall torque to
zero at no-load speed). Replace the numbers in RobotConfig with real motor
data later; the code does not change.
"""
from __future__ import annotations

import hashlib
import math
import os
import tempfile

import numpy as np

from .config import RobotConfig

_HALF_PI = 1.5707963


def _cyl_inertia(m, r, w):
    izz = 0.5 * m * r * r
    ixx = m * (3 * r * r + w * w) / 12.0
    return ixx, ixx, izz


def build_urdf(cfg: RobotConfig) -> str:
    L, T = cfg.wheelbase, cfg.track_width
    cl, cw, ch = cfg.chassis_length, cfg.chassis_width, cfg.chassis_height
    cz = cfg.ground_clearance + ch / 2.0 - cfg.wheel_radius  # chassis centre above axle
    m = cfg.chassis_mass
    bi = (m * (cw ** 2 + ch ** 2) / 12.0, m * (cl ** 2 + ch ** 2) / 12.0, m * (cl ** 2 + cw ** 2) / 12.0)
    wi = _cyl_inertia(cfg.wheel_mass, cfg.wheel_radius, cfg.wheel_width)

    def wheel_link(name):
        return f"""
  <link name="{name}">
    <inertial><mass value="{cfg.wheel_mass}"/>
      <inertia ixx="{wi[0]:.6f}" iyy="{wi[1]:.6f}" izz="{wi[2]:.6f}" ixy="0" ixz="0" iyz="0"/></inertial>
    <collision><geometry><cylinder radius="{cfg.wheel_radius}" length="{cfg.wheel_width}"/></geometry></collision>
    <visual><geometry><cylinder radius="{cfg.wheel_radius}" length="{cfg.wheel_width}"/></geometry>
      <material name="tyre"><color rgba="0.08 0.08 0.08 1"/></material></visual>
  </link>"""

    def steer_link(name):
        return f"""
  <link name="{name}">
    <inertial><mass value="0.05"/>
      <inertia ixx="0.0001" iyy="0.0001" izz="0.0001" ixy="0" ixz="0" iyz="0"/></inertial>
  </link>"""

    def fixed_axle(name, parent, child, x, y):
        return f"""
  <joint name="{name}" type="continuous">
    <parent link="{parent}"/><child link="{child}"/>
    <origin xyz="{x} {y} 0" rpy="{-_HALF_PI} 0 0"/><axis xyz="0 0 1"/>
  </joint>"""

    def steer_joint(name, child, x, y):
        return f"""
  <joint name="{name}" type="revolute">
    <parent link="base_link"/><child link="{child}"/>
    <origin xyz="{x} {y} 0"/><axis xyz="0 0 1"/>
    <limit lower="-0.9" upper="0.9" effort="50" velocity="10"/>
  </joint>"""

    def steered_wheel_joint(name, parent, child):
        return f"""
  <joint name="{name}" type="continuous">
    <parent link="{parent}"/><child link="{child}"/>
    <origin xyz="0 0 0" rpy="{-_HALF_PI} 0 0"/><axis xyz="0 0 1"/>
  </joint>"""

    xml = f"""<?xml version="1.0"?>
<robot name="firebot">
  <link name="base_link">
    <inertial><origin xyz="0 0 {cz}"/><mass value="{m}"/>
      <inertia ixx="{bi[0]:.5f}" iyy="{bi[1]:.5f}" izz="{bi[2]:.5f}" ixy="0" ixz="0" iyz="0"/></inertial>
    <collision><origin xyz="0 0 {cz}"/><geometry><box size="{cl} {cw} {ch}"/></geometry></collision>
    <visual><origin xyz="0 0 {cz}"/><geometry><box size="{cl} {cw} {ch}"/></geometry>
      <material name="body"><color rgba="0.85 0.12 0.10 1"/></material></visual>
  </link>"""
    xml += steer_link("steer_fl") + steer_link("steer_fr")
    xml += wheel_link("wheel_fl") + wheel_link("wheel_fr") + wheel_link("wheel_rl") + wheel_link("wheel_rr")
    xml += steer_joint("j_steer_fl", "steer_fl", L / 2, T / 2)
    xml += steer_joint("j_steer_fr", "steer_fr", L / 2, -T / 2)
    xml += steered_wheel_joint("j_wheel_fl", "steer_fl", "wheel_fl")
    xml += steered_wheel_joint("j_wheel_fr", "steer_fr", "wheel_fr")
    xml += fixed_axle("j_wheel_rl", "base_link", "wheel_rl", -L / 2, T / 2)
    xml += fixed_axle("j_wheel_rr", "base_link", "wheel_rr", -L / 2, -T / 2)
    xml += "\n</robot>\n"
    return xml


def urdf_path(cfg: RobotConfig) -> str:
    xml = build_urdf(cfg)
    h = hashlib.md5(xml.encode()).hexdigest()[:10]
    path = os.path.join(tempfile.gettempdir(), f"firebot_{h}.urdf")
    try:
        if os.path.getsize(path) > 0:
            return path
    except OSError:
        pass

    fd, tmp_path = tempfile.mkstemp(prefix=f"firebot_{h}_", suffix=".urdf", dir=tempfile.gettempdir())
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(xml)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except OSError:
        try:
            if os.path.getsize(path) > 0:
                return path
        except OSError:
            pass
        raise
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
    return path


class FireBotRobot:
    def __init__(self, bc, cfg: RobotConfig, pos_xy, yaw, mass_scale=1.0,
                 friction_scale=1.0, torque_scale=1.0):
        self.bc, self.cfg = bc, cfg
        self.torque_scale = torque_scale
        z = cfg.wheel_radius + 0.005
        orn = bc.getQuaternionFromEuler([0, 0, yaw])
        self.id = bc.loadURDF(urdf_path(cfg), [pos_xy[0], pos_xy[1], z], orn)
        self.joint = {}
        for i in range(bc.getNumJoints(self.id)):
            self.joint[bc.getJointInfo(self.id, i)[1].decode()] = i
        self.wheels = [self.joint[n] for n in ("j_wheel_fl", "j_wheel_fr", "j_wheel_rl", "j_wheel_rr")]
        self.steer = [self.joint["j_steer_fl"], self.joint["j_steer_fr"]]
        self.driven = (self.wheels[2:] if cfg.drive == "rwd" else self.wheels)

        for j in self.wheels:  # free-spinning until we command them
            bc.setJointMotorControl2(self.id, j, bc.VELOCITY_CONTROL, targetVelocity=0, force=0)
            bc.changeDynamics(self.id, j, lateralFriction=cfg.wheel_friction * friction_scale,
                              rollingFriction=0.002, spinningFriction=0.002, linearDamping=0, angularDamping=0)
        bc.changeDynamics(self.id, -1, mass=cfg.chassis_mass * mass_scale, lateralFriction=0.3)

        self.v_cmd = 0.0
        self.steer_cmd = 0.0
        self.steer_actual = 0.0
        self.last_power_w = 0.0

    # --------------------------------------------------------------- control
    def set_command(self, v_target: float, steer_target: float):
        self.v_cmd = float(np.clip(v_target, -self.cfg.max_speed, self.cfg.max_speed))
        self.steer_cmd = float(np.clip(steer_target, -self.cfg.max_steer, self.cfg.max_steer))

    def update_steering(self, dt: float):
        """Slew-limited steering servo with Ackermann geometry (once / control step)."""
        c = self.cfg
        step = c.steer_rate * dt
        self.steer_actual += float(np.clip(self.steer_cmd - self.steer_actual, -step, step))
        d = self.steer_actual
        if abs(d) < 1e-6:
            dl = dr = 0.0
        else:
            t = math.tan(d)
            dl = math.atan2(c.wheelbase * t, c.wheelbase - 0.5 * c.track_width * t)
            dr = math.atan2(c.wheelbase * t, c.wheelbase + 0.5 * c.track_width * t)
        for j, ang in zip(self.steer, (dl, dr)):
            self.bc.setJointMotorControl2(self.id, j, self.bc.POSITION_CONTROL, targetPosition=ang,
                                          force=c.steer_servo_force, positionGain=0.6, velocityGain=0.8)

    def physics_tick(self) -> float:
        """Apply motor torque-speed model for one physics substep; returns electrical power (W)."""
        c = self.cfg
        w_target = self.v_cmd / c.wheel_radius
        states = self.bc.getJointStates(self.id, self.driven)
        forces, p_elec = [], 0.0
        for st in states:
            w = st[1]
            if w * w_target < 0 or abs(w_target) < 1e-3:
                avail = c.motor_stall_torque            # braking / reversing
            else:
                avail = c.motor_stall_torque * max(0.02, 1.0 - abs(w) / c.motor_no_load_speed)
            forces.append(avail * self.torque_scale)
            tau = abs(st[3])
            p_elec += abs(tau * w) / c.motor_efficiency + c.copper_loss_k * tau * tau
        self.bc.setJointMotorControlArray(self.id, self.driven, self.bc.VELOCITY_CONTROL,
                                          targetVelocities=[w_target] * len(self.driven), forces=forces)
        self.last_power_w = p_elec
        return p_elec

    # ----------------------------------------------------------------- state
    def pose(self):
        pos, orn = self.bc.getBasePositionAndOrientation(self.id)
        yaw = self.bc.getEulerFromQuaternion(orn)[2]
        up_z = self.bc.getMatrixFromQuaternion(orn)[8]
        return np.array(pos), yaw, up_z

    def body_velocity(self):
        v, w = self.bc.getBaseVelocity(self.id)
        _, yaw, _ = self.pose()
        c, s = math.cos(yaw), math.sin(yaw)
        vx = c * v[0] + s * v[1]
        vy = -s * v[0] + c * v[1]
        return vx, vy, w[2]
