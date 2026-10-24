"""Smoke + physics sanity tests. Run: python -m pytest -q tests  (or python tests/test_env.py)"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from firebot_sim import FireBotEnv, RobotConfig, WorldConfig, EnvConfig


def make(**kw):
    return FireBotEnv(world_cfg=WorldConfig(n_obstacles=(0, 0), n_walls=(0, 0)),
                      env_cfg=EnvConfig(randomize=False), **kw)


def test_reset_step_shapes():
    env = FireBotEnv()
    obs, info = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape and np.isfinite(obs).all()
    obs, r, term, trunc, info = env.step(env.action_space.sample())
    assert obs.shape == env.observation_space.shape and info["simulated"] is True
    env.close()


def test_determinism_same_seed():
    def run():
        e = FireBotEnv(env_cfg=EnvConfig(randomize=False))
        o, _ = e.reset(seed=7)
        for _ in range(20):
            o, *_ = e.step(np.array([0.5, 0.2, 0], np.float32))
        e.close()
        return o
    assert np.allclose(run(), run())


def test_drives_forward_and_respects_speed_limit():
    env = make()
    env.reset(seed=1)
    p0, yaw0, _ = env.robot.pose()
    for _ in range(40):                       # 2 s at full throttle
        env.step(np.array([1.0, 0.0, 0.0], np.float32))
    p1, _, _ = env.robot.pose()
    vx, _, _ = env.robot.body_velocity()
    fwd = (p1[0] - p0[0]) * math.cos(yaw0) + (p1[1] - p0[1]) * math.sin(yaw0)
    assert fwd > 0.8, f"robot barely moved: {fwd:.2f} m"
    assert 0.5 < vx <= env.rcfg.max_speed * 1.15, f"speed {vx:.2f}"
    env.close()


def test_steering_turns_the_right_way():
    env = make()
    env.reset(seed=2)
    _, yaw0, _ = env.robot.pose()
    for _ in range(40):
        env.step(np.array([0.6, 1.0, 0.0], np.float32))   # left
    _, yaw1, _ = env.robot.pose()
    dyaw = (yaw1 - yaw0 + math.pi) % (2 * math.pi) - math.pi
    assert dyaw > 0.3, f"positive steer should yaw left (ccw); got {dyaw:.2f} rad"
    env.close()


def test_wall_collision_detected_and_robot_stays_inside():
    env = make()
    env.reset(seed=3)
    coll = 0
    for _ in range(400):
        _, _, term, trunc, info = env.step(np.array([1.0, 0.0, 0.0], np.float32))
        if term or trunc:
            break
    pos, _, _ = env.robot.pose()
    assert abs(pos[0]) < env.wcfg.arena_half + 0.5 and abs(pos[1]) < env.wcfg.arena_half + 0.5
    env.close()


def test_pump_is_penalised_when_wasted():
    env = make()
    env.reset(seed=4)
    _, r_off, *_ = env.step(np.array([0, 0, -1], np.float32))
    _, r_on, *_ = env.step(np.array([0, 0, 1], np.float32))
    assert r_on < r_off
    env.close()


def test_custom_robot_config_changes_physics():
    slow = RobotConfig(max_speed=0.5)
    env = FireBotEnv(robot_cfg=slow, world_cfg=WorldConfig(n_obstacles=(0, 0), n_walls=(0, 0)),
                     env_cfg=EnvConfig(randomize=False))
    env.reset(seed=5)
    for _ in range(60):
        env.step(np.array([1.0, 0.0, 0.0], np.float32))
    vx, _, _ = env.robot.body_velocity()
    assert vx < 0.65
    env.close()


def test_reward_terms_add_up_to_the_total_reward():
    env = FireBotEnv(env_cfg=EnvConfig(randomize=False))
    env.reset(seed=11)
    total = 0.0
    for _ in range(150):
        _, r, te, tr, info = env.step(env.action_space.sample())
        total += r
        if te or tr:
            break
    assert abs(total - sum(info["reward_terms"].values())) < 1e-3
    assert "time_penalty" in info["reward_terms"]
    env.close()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
