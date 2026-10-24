"""Evaluate a trained policy or the oracle baseline.

  python scripts/evaluate.py --baseline --episodes 20
  python scripts/evaluate.py --model runs/firebot_ppo/final.zip --episodes 50
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from firebot_sim import FireBotEnv, RobotConfig, WorldConfig
from firebot_sim.baseline import OraclePlanner

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=None)
ap.add_argument("--vecnorm", default=None)
ap.add_argument("--baseline", action="store_true")
ap.add_argument("--episodes", type=int, default=20)
ap.add_argument("--robot-config", default=None)
ap.add_argument("--easy", action="store_true")
a = ap.parse_args()

rc = RobotConfig.from_json(a.robot_config) if a.robot_config else RobotConfig()
wc = WorldConfig(n_people=(1, 1), n_fires=(1, 1), n_obstacles=(2, 4), n_walls=(0, 0)) if a.easy else WorldConfig()
env = FireBotEnv(rc, wc)
model = vn = None
if a.model:
    from stable_baselines3 import PPO
    model = PPO.load(a.model)
    if a.vecnorm:
        import pickle
        with open(a.vecnorm, "rb") as f:
            vn = pickle.load(f)

stats = {"success": 0, "collisions": [], "steps": [], "reasons": {}}
for ep in range(a.episodes):
    obs, info = env.reset(seed=1000 + ep)
    planner = OraclePlanner(env) if a.baseline else None
    done = False
    while not done:
        if planner:
            act = planner.act()
        else:
            o = vn.normalize_obs(obs) if vn else obs
            act, _ = model.predict(o, deterministic=True)
        obs, r, term, trunc, info = env.step(act)
        done = term or trunc
    stats["success"] += info["end_reason"] == "success"
    stats["collisions"].append(info["collisions"])
    stats["steps"].append(info["steps"])
    stats["reasons"][info["end_reason"]] = stats["reasons"].get(info["end_reason"], 0) + 1
    print(f"ep {ep:3d} {info['end_reason']:<17} people {info['people_located']}/{info['n_people']} "
          f"fires {info['fires_out']}/{info['n_fires']} coll {info['collisions']:3d} steps {info['steps']}")
print(f"\nsuccess {stats['success']}/{a.episodes}  mean collisions {np.mean(stats['collisions']):.1f}  "
      f"mean steps {np.mean(stats['steps']):.0f}  reasons {stats['reasons']}")
