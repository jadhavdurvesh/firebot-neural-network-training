"""Brain tests: observation equivalence sim<->robot code path, and safety supervisor rules."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from firebot_sim import FireBotEnv, RobotConfig, WorldConfig, EnvConfig
from firebot_sim.baseline import OraclePlanner
from firebot_brain import FireBotBrain, SafetySupervisor, SensorFrame, Command
from firebot_brain.sim_hal import SimHAL


def _frame(**kw):
    cfg = RobotConfig()
    base = dict(t=0.0, x=0, y=0, yaw=0, vx=0, vy=0, wz=0, steer_angle=0,
                lidar_m=np.full(cfg.lidar_rays, cfg.lidar_range, np.float32))
    base.update(kw)
    return SensorFrame(**base)


def test_obs_identical_between_sim_and_robot_code_path():
    """Brain (robot-side) must build the same 104-float observation the env gave the policy in training."""
    rc = RobotConfig(detection_prob=1.0)
    worst = 0.0
    cases = [(WorldConfig(n_people=(1, 2)), 1), (WorldConfig(n_people=(1, 2)), 2),
             (WorldConfig(layout="building", n_people=(1, 2)), 4)]
    for wcfg, seed in cases:
        env = FireBotEnv(rc, wcfg, EnvConfig(randomize=False))
        hal = SimHAL(env)
        hal.reset(seed=seed)
        brain = FireBotBrain(None, rc, n_people=env.n_people, n_fires=env.n_fires, safety=False)
        planner = OraclePlanner(env)
        for _ in range(900):
            act = planner.act()
            obs, r, term, trunc, info = hal.send(_cmd(act))
            brain._last = np.clip(act, -1, 1).astype(np.float32)
            brain.mem.update(hal.read())
            b = brain.build_obs(hal.read())
            worst = max(worst, float(np.abs(b - obs).max()))
            if term or trunc:
                break
        env.close()
    assert worst < 1e-5, f"obs mismatch {worst}"


def _cmd(a):
    return Command(float(a[0]), float(a[1]), float(a[2]))


def test_safety_blocks_forward_motion_into_wall():
    s = SafetySupervisor()
    lid = np.full(24, 6.0, np.float32)
    lid[0] = 0.30                                         # wall dead ahead
    cmd, why = s.filter(_frame(lidar_m=lid), Command(1.0, 0.0, -1))
    assert cmd.throttle == 0.0 and "obstacle_limit" in why
    cmd, _ = s.filter(_frame(lidar_m=lid), Command(-0.5, 0.0, -1))     # reversing allowed
    assert cmd.throttle == -0.5


def test_safety_stops_on_estop_override_thermal_battery_stale():
    s = SafetySupervisor()
    for kw, name in ((dict(estop=True), "estop"), (dict(manual_override=True), "manual_override"),
                     (dict(temp_c=95.0), "thermal_critical"), (dict(battery_frac=0.02), "battery_critical")):
        cmd, why = s.filter(_frame(**kw), Command(1, 1, 1))
        assert (cmd.throttle, cmd.steer) == (0.0, 0.0) and cmd.pump < 0 and name in why, (name, why)
    cmd, why = s.filter(_frame(t=0.0), Command(1, 0, 1), now=1.0)
    assert "stale_sensors" in why and cmd.throttle == 0.0


def test_safety_limits_speed_when_hot_or_low_battery_and_blocks_dry_pump():
    s = SafetySupervisor()
    cmd, why = s.filter(_frame(temp_c=75.0), Command(1.0, 0, -1))
    assert cmd.throttle <= 0.5 and "thermal_warn" in why
    cmd, why = s.filter(_frame(battery_frac=0.10), Command(1.0, 0, -1))
    assert cmd.throttle <= 0.6 and "battery_low" in why
    cmd, why = s.filter(_frame(water_frac=0.0), Command(0, 0, 1.0))
    assert cmd.pump < 0 and "pump_no_water" in why


def test_nan_command_is_stopped():
    cmd, why = SafetySupervisor().filter(_frame(), Command(float("nan"), 0, 0))
    assert cmd.throttle == 0.0 and "non_finite" in why


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
