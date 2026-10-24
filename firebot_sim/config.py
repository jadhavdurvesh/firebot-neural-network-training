"""Configuration for the FireBot physics training simulation.

EVERY physical number in RobotConfig is an ASSUMPTION until real hardware is
chosen (see firebot-core/DECISIONS.md: voltages, motors, battery undecided).
When hardware is finalised: measure / read datasheets, write the values into a
JSON file (see configs/robot_assumed.json) and pass it with --robot-config.
Nothing else in the codebase hardcodes robot physics.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from typing import Tuple


@dataclass
class RobotConfig:
    # ---- chassis geometry (m, kg) -- ASSUMED -------------------------------
    chassis_length: float = 0.60
    chassis_width: float = 0.38
    chassis_height: float = 0.16
    chassis_mass: float = 12.0
    ground_clearance: float = 0.05
    wheelbase: float = 0.46           # front-axle to rear-axle
    track_width: float = 0.44         # left-wheel to right-wheel
    wheel_radius: float = 0.09
    wheel_width: float = 0.06
    wheel_mass: float = 0.6
    wheel_friction: float = 1.0

    # ---- drivetrain (car-like / Ackermann) -- ASSUMED ----------------------
    drive: str = "rwd"                # "rwd" or "awd"
    motor_stall_torque: float = 3.0   # N*m at the wheel, per driven wheel
    motor_no_load_speed: float = 25.0 # rad/s at the wheel (-> ~2.25 m/s)
    motor_efficiency: float = 0.75
    copper_loss_k: float = 1.5        # W per (N*m)^2, crude I^2R term
    max_speed: float = 1.8            # m/s speed-controller limit
    max_steer: float = 0.5            # rad, virtual centre-wheel steering angle
    steer_rate: float = 2.5           # rad/s steering-servo slew rate
    steer_servo_force: float = 30.0   # N*m

    # ---- sensing -- ASSUMED -------------------------------------------------
    lidar_rays: int = 24
    lidar_range: float = 6.0
    lidar_noise_std: float = 0.02     # m
    lidar_dropout: float = 0.01
    camera_fov_deg: float = 90.0
    camera_mount: Tuple[float, float, float] = (0.36, 0.0, 0.30)
    detection_range: float = 8.0
    detection_prob: float = 0.97      # per-step prob. of detecting a visible target
    # forward IR flame/thermal strip (like a flame-sensor array or one row of a thermal camera) -- ASSUMED
    ir_bins: int = 16                 # horizontal bins across the field of view
    ir_fov_deg: float = 120.0
    ir_range: float = 10.0            # m; NOT degraded by smoke (unlike lidar/camera), blocked by walls
    ir_noise_std: float = 0.01

    # ---- power / thermal / pump -- ASSUMED, synthetic ----------------------
    battery_wh: float = 12.0
    idle_power_w: float = 25.0        # compute + sensors
    energy_time_scale: float = 5.0    # accelerates drain so battery matters in
                                      # short episodes (training pressure only)
    pump_power_w: float = 20.0
    pump_water_rate: float = 0.02     # tank fraction / s while spraying
    pump_range: float = 2.5           # m
    pump_half_angle: float = 0.6      # rad
    pump_suppress_rate: float = 0.25  # fire intensity / s while hit
    thermal_ambient_c: float = 28.0
    thermal_capacity: float = 600.0   # J/degC (illustrative)
    thermal_cooling_k: float = 0.02   # 1/s at 100% cooling effectiveness
    thermal_warn_c: float = 70.0      # illustrative, NOT a component rating
    thermal_critical_c: float = 90.0  # illustrative, NOT a component rating
    fire_heat_rate: float = 1.0       # degC/s at zero distance, intensity 1
    fire_heat_radius: float = 6.0     # m

    @property
    def robot_radius(self) -> float:
        return 0.5 * (self.chassis_length ** 2 + self.track_width ** 2) ** 0.5 + 0.03

    @classmethod
    def from_json(cls, path: str) -> "RobotConfig":
        with open(path) as f:
            data = json.load(f)
        names = {f_.name for f_ in fields(cls)}
        unknown = set(data) - names
        if unknown:
            raise ValueError(f"unknown RobotConfig keys: {sorted(unknown)}")
        if "camera_mount" in data:
            data["camera_mount"] = tuple(data["camera_mount"])
        return cls(**data)

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)


@dataclass
class WorldConfig:
    arena_half: float = 6.0                 # arena is 2*arena_half square (m)
    n_people: Tuple[int, int] = (1, 2)      # inclusive range
    n_fires: Tuple[int, int] = (1, 1)
    n_obstacles: Tuple[int, int] = (3, 7)
    n_walls: Tuple[int, int] = (0, 2)
    cell: float = 0.15                      # planning-grid resolution (m)
    person_radius: float = 0.2
    fire_radius: float = 0.35
    cooling_failure_prob: float = 0.25      # episode has a mid-run cooling fault
    layout: str = "open"                    # "open" (obstacles in one space) or "building" (rooms + hallway)
    n_rooms_per_side: Tuple[int, int] = (2, 3)   # building: rooms on each side of the hallway
    hall_width: float = 2.0                 # building: hallway width (m)
    door_width: float = 1.5                 # building: doorway width (m)
    room_furniture: Tuple[int, int] = (0, 2)     # building: boxes per room


@dataclass
class EnvConfig:
    physics_hz: int = 120
    control_hz: int = 20
    max_steps: int = 2000
    randomize: bool = True                  # domain randomisation on reset
    max_action_delay: int = 1               # control steps of actuator latency
    coverage_grid: int = 6                  # NxN exploration memory in obs
    locate_dist: float = 1.8
    # reward weights
    r_time: float = -0.01
    r_explore: float = 0.05
    r_detect: float = 1.0
    r_locate: float = 20.0
    r_fire_out: float = 20.0
    r_success: float = 50.0
    r_progress: float = 0.5                 # per metre closer to known goal
    r_collision: float = -0.5
    r_wasted_pump: float = -0.03
    r_thermal: float = -0.02
    r_fail: float = -20.0


# Domain randomisation applied at every reset when EnvConfig.randomize is True.
# env.py reads these values, and every run record stores them, so the paper's "randomisation table" is exact.
DOMAIN_RANDOMIZATION = {
    "chassis_mass_scale": (0.85, 1.15),
    "wheel_friction_scale": (0.7, 1.2),
    "motor_torque_scale": (0.85, 1.1),
    "lidar_noise_scale": (0.5, 1.5),
    "pose_noise_std": {"position_m": 0.03, "yaw_rad": 0.02},
    "initial_battery_fraction": (0.5, 1.0),
    "ambient_temp_offset_c": (-5.0, 8.0),
    "smoke_level": (0.0, 0.7),
    "action_delay_steps": "integer in [0, EnvConfig.max_action_delay]",
    "cooling_failure": "probability WorldConfig.cooling_failure_prob; starts at a random step in [150, max_steps/2]; "
                       "cooling effectiveness drops to 0.15",
}
