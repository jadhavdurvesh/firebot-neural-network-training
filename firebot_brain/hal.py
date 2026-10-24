"""Hardware abstraction layer (HAL): the ONLY thing you implement for the real robot.

The brain consumes a SensorFrame and returns a Command. Everything between
(memory, observation vector, neural policy, safety supervisor) is hardware-agnostic
and identical to what ran in simulation.

Conventions (must match the simulator):
  * world frame: x forward-right-hand ENU style, yaw in rad, counter-clockwise positive, 0 = +x
  * lidar_m[k] is the range (m) at angle yaw + 2*pi*k/N  (k=0 straight ahead, increasing counter-clockwise);
    use `lidar_range` where there is no return
  * ir_bins[k]: heat reading for bearing -fov/2 + (k+0.5)*fov/N relative to the robot's heading (negative = right,
    positive = left, 0 = straight ahead), N = RobotConfig.ir_bins, fov = RobotConfig.ir_fov_deg
  * throttle in [-1, 1] maps to [-max_speed, +max_speed] m/s (speed setpoint for your motor controller);
    steer in [-1, 1] maps to [-max_steer, +max_steer] rad (virtual centre-wheel Ackermann angle)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Protocol

import numpy as np


@dataclass
class Detection:
    cls: str                 # "person" or "fire"
    x: float                 # world-frame position (m), from vision + mapping
    y: float
    confidence: float = 1.0
    active: bool = True      # fires only: False once the fire is confirmed out (e.g. thermal camera)


@dataclass
class SensorFrame:
    t: float                         # timestamp, seconds (monotonic)
    x: float                         # pose estimate from firebot-mapping / odometry
    y: float
    yaw: float
    vx: float                        # body-frame velocities (m/s, rad/s)
    vy: float
    wz: float
    steer_angle: float               # measured steering angle (rad)
    lidar_m: np.ndarray              # (N,) ranges in metres
    detections: List[Detection] = field(default_factory=list)   # CURRENTLY visible targets
    ir_bins: np.ndarray = field(default_factory=lambda: np.zeros(16, np.float32))  # forward IR strip, left->right
    battery_frac: float = 1.0
    water_frac: float = 1.0
    temp_c: float = 25.0             # hottest monitored electronics zone
    ambient_c: float = 25.0
    cooling_eff: float = 1.0         # 0..1, from cooling-system health
    estop: bool = False              # hardware e-stop pressed
    manual_override: bool = False    # human has taken control


@dataclass
class Command:
    throttle: float = 0.0
    steer: float = 0.0
    pump: float = -1.0               # > 0 = pump on


class SensorHAL(Protocol):
    def read(self) -> SensorFrame: ...


class ActuatorHAL(Protocol):
    def send(self, cmd: Command) -> None: ...
    def stop(self) -> None: ...
