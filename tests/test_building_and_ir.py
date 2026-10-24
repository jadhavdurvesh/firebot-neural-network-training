"""Building layout (rooms + hallway) and the IR fire sensor."""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from firebot_sim import EnvConfig, FireBotEnv, RobotConfig, WorldConfig
from firebot_sim.world import generate_world


def test_buildings_have_rooms_hallway_and_reachable_targets():
    rr = RobotConfig().robot_radius
    wc = WorldConfig(layout="building", n_people=(2, 4), n_fires=(1, 2))
    rng = np.random.default_rng(0)
    for _ in range(25):
        L = generate_world(rng, wc, rr)            # raises if a target were unreachable
        assert len(L.rooms) >= 2 and any(b.h > 0.3 for b in L.boxes)
        assert 2 <= len(L.people) <= 4 and 1 <= len(L.fires) <= 2
        for (x, y) in L.people + L.fires:          # every target is inside some room
            assert any(x0 < x < x1 and y0 < y < y1 for x0, x1, y0, y1 in L.rooms)
        sx, sy, _ = L.spawn                        # robot starts in the hallway, not in a room
        assert not any(x0 < sx < x1 and y0 < sy < y1 for x0, x1, y0, y1 in L.rooms)


def _teleport_facing_fire(env, dist=3.0):
    fx, fy = env.layout.fires[0]
    n = math.hypot(fx, fy)
    ux, uy = (fx / n, fy / n) if n > 0.5 else (1.0, 0.0)          # step from the fire towards the arena centre
    rx, ry = fx - ux * dist, fy - uy * dist
    yaw = math.atan2(fy - ry, fx - rx)
    env.bc.resetBasePositionAndOrientation(env.robot.id, [rx, ry, 0.095], env.bc.getQuaternionFromEuler([0, 0, yaw]))
    return rx, ry, yaw, (fx, fy)


def _open_env():
    return FireBotEnv(world_cfg=WorldConfig(n_people=(1, 1), n_fires=(1, 1), n_obstacles=(0, 0), n_walls=(0, 0)),
                      env_cfg=EnvConfig(randomize=False))


def test_ir_points_at_fire_and_is_blind_behind_the_robot_or_a_wall():
    env = _open_env()
    env.reset(seed=3)
    rx, ry, yaw, (fx, fy) = _teleport_facing_fire(env)
    env._sense()
    ir = env._ir.copy()
    nb = len(ir)
    assert ir.max() > 0.1 and abs(int(ir.argmax()) - (nb - 1) / 2) <= 2      # strong, roughly dead ahead
    # facing away: nothing in the field of view
    env.bc.resetBasePositionAndOrientation(env.robot.id, [rx, ry, 0.095], env.bc.getQuaternionFromEuler([0, 0, yaw + math.pi]))
    env._sense()
    assert env._ir.max() < 0.05
    # back to facing the fire but put a tall wall in between: IR is blocked
    env.bc.resetBasePositionAndOrientation(env.robot.id, [rx, ry, 0.095], env.bc.getQuaternionFromEuler([0, 0, yaw]))
    env._static_box((rx + fx) / 2, (ry + fy) / 2, 0.1, 2.0, yaw, hz=0.6)
    env._sense()
    assert env._ir.max() < 0.05
    env.close()


def test_smoke_degrades_camera_range_but_not_ir():
    env = _open_env()
    env.reset(seed=3)
    _teleport_facing_fire(env)
    env._sense()
    clear = env._ir.copy()
    env.smoke = 0.7
    env._sense()
    assert abs(env._ir.max() - clear.max()) < 0.05
    env.close()
