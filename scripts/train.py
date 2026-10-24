"""Train a PPO policy in the FireBot physics sim.

  python scripts/train.py --steps 2_000_000 --n-envs 8
  python scripts/train.py --robot-config configs/robot_assumed.json
Requires: pip install stable-baselines3 torch
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv

from firebot_sim import FireBotEnv, RobotConfig, WorldConfig


def make(robot_cfg_path, easy):
    rc = RobotConfig.from_json(robot_cfg_path) if robot_cfg_path else RobotConfig()
    wc = WorldConfig(n_people=(1, 1), n_fires=(1, 1), n_obstacles=(2, 4), n_walls=(0, 0)) if easy else WorldConfig()
    return lambda: FireBotEnv(rc, wc)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2_000_000)
    ap.add_argument("--n-envs", type=int, default=4)
    ap.add_argument("--robot-config", default=None)
    ap.add_argument("--easy", action="store_true", help="1 person, 1 fire, few obstacles (curriculum stage 1)")
    ap.add_argument("--out", default="runs/firebot_ppo")
    ap.add_argument("--resume", default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    vec_cls = SubprocVecEnv if a.n_envs > 1 else DummyVecEnv
    venv = make_vec_env(make(a.robot_config, a.easy), n_envs=a.n_envs, vec_env_cls=vec_cls)
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)
    model = PPO("MlpPolicy", venv, n_steps=1024, batch_size=256, gamma=0.995, gae_lambda=0.95,
                learning_rate=3e-4, ent_coef=0.005, policy_kwargs=dict(net_arch=[256, 256]),
                tensorboard_log=None, verbose=1) if not a.resume else PPO.load(a.resume, env=venv)
    ckpt = CheckpointCallback(save_freq=max(50_000 // a.n_envs, 1), save_path=a.out, name_prefix="ppo")
    model.learn(a.steps, callback=ckpt, progress_bar=False)
    model.save(os.path.join(a.out, "final"))
    venv.save(os.path.join(a.out, "vecnormalize.pkl"))
    print("saved to", a.out)
