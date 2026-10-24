"""Teacher -> student distillation (DAgger) followed by PPO fine-tuning.

  python scripts/distill_then_ppo.py --out runs/brain_v2 --ppo-minutes 25

1. A reactive scripted teacher (uses only student-observable information) labels states.
2. DAgger: the student drives (mixed with the teacher at decreasing beta), the teacher relabels, the
   dataset aggregates, the student (an SB3 PPO actor-critic MLP) is re-fit by supervised learning.
3. PPO fine-tunes the distilled student on mixed-difficulty worlds with smoke / latency / friction
   randomisation, keeping the best checkpoint by rolling success rate.
"""
import argparse
import json
import os
import sys
import time
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from firebot_sim import FireBotEnv, RobotConfig, WorldConfig
from firebot_sim.teacher import ReactiveTeacher

STAGES = [
    WorldConfig(n_people=(1, 1), n_fires=(1, 1), n_obstacles=(0, 2), n_walls=(0, 0)),
    WorldConfig(n_people=(1, 1), n_fires=(1, 1), n_obstacles=(2, 5), n_walls=(0, 1)),
    WorldConfig(),
]


def student_action(model, obs_rms, obs):
    x = np.clip((obs - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8), -10, 10).astype(np.float32)
    with torch.no_grad():
        mean = model.policy.get_distribution(torch.as_tensor(x)[None]).distribution.mean[0].numpy()
    return np.clip(mean, -1, 1).astype(np.float32)


def rollout(env, model, obs_rms, beta, n_eps, seed0, rng, record=True):
    X, Y, wins = [], [], 0
    for k in range(n_eps):
        obs, _ = env.reset(seed=seed0 + k)
        teacher = ReactiveTeacher(env)
        done = False
        while not done:
            t_act = teacher.act()
            if record:
                X.append(obs.copy()), Y.append(t_act.copy())
            a = t_act if (model is None or rng.random() < beta) else student_action(model, obs_rms, obs)
            obs, _, te, tr, info = env.step(a)
            done = te or tr
        wins += info["end_reason"] == "success"
    return X, Y, wins / n_eps


def train_bc(model, obs_rms, X, Y, epochs=25, lr=1e-3, bs=512):
    Xn = np.clip((np.asarray(X) - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8), -10, 10).astype(np.float32)
    Xt, Yt = torch.as_tensor(Xn), torch.as_tensor(np.asarray(Y, np.float32))
    w = torch.tensor([1.0, 1.0, 2.0])
    opt = torch.optim.Adam(model.policy.parameters(), lr=lr)
    n = len(Xt)
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            out = model.policy.get_distribution(Xt[idx]).distribution.mean
            loss = (F.smooth_l1_loss(out, Yt[idx], reduction="none") * w).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(idx)
    return tot / n


class Progress(BaseCallback):
    def __init__(self, out, minutes, every=20_000):
        super().__init__()
        self.out, self.deadline, self.every = out, time.time() + minutes * 60, every
        self.t0, self.last, self.best = time.time(), 0, -1.0
        self.ep = deque(maxlen=100)

    def _on_step(self):
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if done:
                self.ep.append((info["end_reason"] == "success", info["people_located"] / max(1, info["n_people"]),
                                info["fires_out"] / max(1, info["n_fires"]), info["collisions"]))
        if self.num_timesteps - self.last >= self.every and len(self.ep) >= 30:
            self.last = self.num_timesteps
            a = np.array(self.ep, float)
            print(f"[ppo] t={self.num_timesteps:>8} {time.time()-self.t0:6.0f}s success {a[:,0].mean():.2f} "
                  f"people {a[:,1].mean():.2f} fires {a[:,2].mean():.2f} coll/ep {a[:,3].mean():.0f}", flush=True)
            self.model.save(os.path.join(self.out, "latest_ppo"))
            self.model.get_vec_normalize_env().save(os.path.join(self.out, "latest_ppo_vecnorm.pkl"))
            if a[:, 0].mean() > self.best:
                self.best = a[:, 0].mean()
                self.model.save(os.path.join(self.out, "best_ppo"))
                self.model.get_vec_normalize_env().save(os.path.join(self.out, "best_ppo_vecnorm.pkl"))
        return time.time() < self.deadline


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/brain_v2")
    ap.add_argument("--dagger-eps", type=int, nargs="+", default=[40, 25, 25, 25])
    ap.add_argument("--ppo-minutes", type=float, default=25)
    ap.add_argument("--robot-config", default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rc = RobotConfig.from_json(a.robot_config) if a.robot_config else RobotConfig()
    rng = np.random.default_rng(0)

    env = FireBotEnv(rc, STAGES)                       # mixed difficulty
    eval_env = FireBotEnv(rc, STAGES[2])               # full-difficulty evaluation
    venv = VecNormalize(DummyVecEnv([lambda: FireBotEnv(rc, STAGES)]), norm_obs=True, norm_reward=True, clip_obs=10.0)
    model = PPO("MlpPolicy", venv, n_steps=2048, batch_size=256, n_epochs=8, gamma=0.995, gae_lambda=0.95,
                learning_rate=3e-4, policy_kwargs=dict(net_arch=[256, 256]), verbose=0)

    # ---- DAgger (resumable: state saved after every iteration)
    t0 = time.time()
    state_path = os.path.join(a.out, "dagger_state.pt")
    betas = [0.5, 0.3, 0.1, 0.0]
    if os.path.exists(state_path):
        st = torch.load(state_path, weights_only=False)
        X, Y, start = st["X"], st["Y"], st["it"] + 1
        model.policy.load_state_dict(st["policy"])
        venv.obs_rms.mean, venv.obs_rms.var, venv.obs_rms.count = st["mean"], st["var"], 1e4
        print(f"[dagger] resumed after iter{st['it']} with {len(X)} samples", flush=True)
    else:
        X, Y, tw = rollout(env, None, None, 1.0, a.dagger_eps[0], 10_000, rng)
        print(f"[dagger] iter0 teacher data {len(X)} samples, teacher success {tw:.2f} ({time.time()-t0:.0f}s)", flush=True)
        Xa = np.asarray(X)
        venv.obs_rms.mean, venv.obs_rms.var, venv.obs_rms.count = Xa.mean(0), Xa.var(0) + 1e-6, 1e4
        loss = train_bc(model, venv.obs_rms, X, Y)
        print(f"[dagger] iter0 bc loss {loss:.4f}", flush=True)
        start = 1
        torch.save(dict(X=X, Y=Y, it=0, policy=model.policy.state_dict(), mean=venv.obs_rms.mean, var=venv.obs_rms.var), state_path)
    for it in range(start, len(a.dagger_eps)):
        n_eps = a.dagger_eps[it]
        Xn, Yn, sw = rollout(env, model, venv.obs_rms, betas[min(it - 1, len(betas) - 1)], n_eps, 20_000 + it * 100, rng)
        X += Xn
        Y += Yn
        loss = train_bc(model, venv.obs_rms, X, Y)
        _, _, ev = rollout(eval_env, model, venv.obs_rms, 0.0, 6, 90_000, rng, record=False)
        print(f"[dagger] iter{it} data {len(X)} bc loss {loss:.4f} student-mix success {sw:.2f} "
              f"| pure student full-world success {ev:.2f} ({time.time()-t0:.0f}s)", flush=True)
        torch.save(dict(X=X, Y=Y, it=it, policy=model.policy.state_dict(), mean=venv.obs_rms.mean, var=venv.obs_rms.var), state_path)
    model.save(os.path.join(a.out, "student_dagger"))
    venv.save(os.path.join(a.out, "student_dagger_vecnorm.pkl"))

    # ---- PPO fine-tune
    model.policy.log_std.data.fill_(-1.2)
    model.learning_rate = 1e-4
    model.lr_schedule = lambda _: 1e-4
    model.clip_range = lambda _: 0.1
    model.ent_coef = 0.0
    print(f"=== PPO fine-tune for {a.ppo_minutes} min ===", flush=True)
    model.learn(10_000_000, callback=Progress(a.out, a.ppo_minutes), reset_num_timesteps=False)
    model.save(os.path.join(a.out, "final"))
    venv.save(os.path.join(a.out, "vecnormalize.pkl"))
    json.dump({"obs_dim": int(model.observation_space.shape[0])}, open(os.path.join(a.out, "meta.json"), "w"))
    print("DONE", flush=True)
