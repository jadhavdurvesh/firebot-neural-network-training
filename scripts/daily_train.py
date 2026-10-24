"""Scheduled reinforcement-learning run (GitHub Actions executes this several times a day).

Isaac-Lab-style loop, on a schedule:
  1. load yesterday's trainer checkpoint (the model is "carried" into today's new simulation)
  2. build TODAY'S worlds: new random seed (= date), parallel envs, domain randomisation,
     difficulty from the curriculum level (current level + the one below, so nothing is forgotten)
  3. PPO: the robot earns/loses named reward points (see EnvConfig / info["reward_terms"])
  4. evaluate challenger vs champion on a FIXED benchmark (same seeds every day -> comparable)
  5. promote the challenger to champion only if it scores better; export champion ONNX
  6. level up the curriculum when the gate success rate is reached
  7. write state + report (GitHub job summary) so tomorrow's run can continue

  python scripts/daily_train.py --state-dir state --minutes 120
"""
import argparse
import csv
import datetime as dt
import json
import os
import pickle
import shutil
import sys
import time
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from firebot_sim import FireBotEnv, RobotConfig, WorldConfig
from firebot_sim.export import export_policy_onnx

LEVELS = [
    ("L0_easy", WorldConfig(n_people=(1, 1), n_fires=(1, 1), n_obstacles=(0, 2), n_walls=(0, 0))),
    ("L1_medium", WorldConfig(n_people=(1, 1), n_fires=(1, 1), n_obstacles=(2, 5), n_walls=(0, 1))),
    ("L2_full", WorldConfig()),
    ("L3_hard", WorldConfig(n_people=(2, 3), n_fires=(1, 2), n_obstacles=(5, 9), n_walls=(1, 3))),
]
EVAL_SEED0 = 100_000          # fixed benchmark seeds: identical every day


def score(m):
    """Single number used to decide promotion (higher is better)."""
    return m["success"] + 0.3 * m["people_rate"] + 0.3 * m["fire_rate"] - 0.0005 * m["collisions_per_ep"]


def evaluate(model, vecnorm_path, rc, wc, episodes):
    vn = pickle.load(open(vecnorm_path, "rb")) if vecnorm_path and os.path.exists(vecnorm_path) else None
    env = FireBotEnv(rc, wc)
    succ, ppl, fir, coll, steps, ret, reasons, terms = [], [], [], [], [], [], {}, {}
    for k in range(episodes):
        obs, _ = env.reset(seed=EVAL_SEED0 + k)
        done, total = False, 0.0
        while not done:
            o = vn.normalize_obs(obs) if vn is not None else obs
            act, _ = model.predict(o, deterministic=True)
            obs, r, te, tr, info = env.step(act)
            total += r
            done = te or tr
        succ.append(info["end_reason"] == "success")
        ppl.append(info["people_located"] / max(1, info["n_people"]))
        fir.append(info["fires_out"] / max(1, info["n_fires"]))
        coll.append(info["collisions"]), steps.append(info["steps"]), ret.append(total)
        reasons[info["end_reason"]] = reasons.get(info["end_reason"], 0) + 1
        for t, v in info["reward_terms"].items():
            terms[t] = terms.get(t, 0.0) + v / episodes
    env.close()
    return {"success": float(np.mean(succ)), "people_rate": float(np.mean(ppl)), "fire_rate": float(np.mean(fir)),
            "collisions_per_ep": float(np.mean(coll)), "mean_steps": float(np.mean(steps)),
            "mean_return": float(np.mean(ret)), "episodes": episodes, "end_reasons": reasons,
            "reward_points_per_episode": {k: round(v, 2) for k, v in sorted(terms.items(), key=lambda kv: -abs(kv[1]))}}


class DailyCallback(BaseCallback):
    def __init__(self, deadline, csv_path, ckpt_fn, every=20_000, ckpt_every_s=900):
        super().__init__()
        self.deadline, self.every, self.ckpt_fn, self.ckpt_every_s = deadline, every, ckpt_fn, ckpt_every_s
        self.t0 = self.t_ckpt = time.time()
        self.last = 0
        self.ep = deque(maxlen=200)
        self.f = open(csv_path, "a", newline="")
        self.w = csv.writer(self.f)
        if self.f.tell() == 0:
            self.w.writerow(["unix_time", "timesteps", "success", "people", "fires", "collisions_per_ep", "episodes"])

    def _on_step(self):
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if done:
                self.ep.append((info["end_reason"] == "success", info["people_located"] / max(1, info["n_people"]),
                                info["fires_out"] / max(1, info["n_fires"]), info["collisions"]))
        if self.num_timesteps - self.last >= self.every and len(self.ep) >= 20:
            self.last = self.num_timesteps
            a = np.array(self.ep, float)
            print(f"[train] t={self.num_timesteps:>9} {time.time()-self.t0:6.0f}s success {a[:,0].mean():.2f} "
                  f"people {a[:,1].mean():.2f} fires {a[:,2].mean():.2f} coll/ep {a[:,3].mean():.0f}", flush=True)
            self.w.writerow([int(time.time()), self.num_timesteps, *a.mean(0).round(4), len(a)])
            self.f.flush()
        if time.time() - self.t_ckpt > self.ckpt_every_s:       # survive cancelled / timed-out runs
            self.t_ckpt = time.time()
            self.ckpt_fn()
        return time.time() < self.deadline

    def _on_training_end(self):
        self.f.close()


def make_raw_venv(rc, wcs, n_envs, base_seed):
    fns = [lambda: FireBotEnv(rc, wcs) for _ in range(n_envs)]
    venv = SubprocVecEnv(fns) if n_envs > 1 else DummyVecEnv(fns)
    venv.seed(base_seed)
    return venv


def load_state(path):
    if os.path.exists(path):
        return json.load(open(path))
    return {"level": 0, "total_steps": 0, "runs": 0, "bad_runs": 0, "champion": None, "history": []}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state-dir", default="state")
    ap.add_argument("--minutes", type=float, default=120)
    ap.add_argument("--n-envs", type=int, default=0, help="0 = number of CPU cores")
    ap.add_argument("--eval-episodes", type=int, default=30)
    ap.add_argument("--date", default=None, help="YYYY-MM-DD; sets today's world seed")
    ap.add_argument("--promote-level-at", type=float, default=0.8)
    ap.add_argument("--robot-config", default=None)
    a = ap.parse_args()

    os.makedirs(a.state_dir, exist_ok=True)
    S = lambda n: os.path.join(a.state_dir, n)
    st = load_state(S("state.json"))
    if a.date:                                       # tests / reproducible runs
        day, today = a.date, f"{a.date} #{st['runs'] + 1}"
    else:
        now = dt.datetime.now(dt.timezone.utc)
        day, today = now.date().isoformat(), now.strftime("%Y-%m-%d %H:%MZ")
    # new worlds EVERY run (not just every day): seed mixes the date with the run counter
    base_seed = (int(day.replace("-", "")) * 1000 + st["runs"]) % (2 ** 31 - 1)
    n_envs = a.n_envs or os.cpu_count() or 1
    rc = RobotConfig.from_json(a.robot_config) if a.robot_config else RobotConfig()
    level = st["level"]
    train_cfgs = [LEVELS[i][1] for i in range(max(0, level - 1), level + 1)]
    print(f"=== {today} | seed {base_seed} | level {LEVELS[level][0]} | train worlds: "
          f"{[LEVELS[i][0] for i in range(max(0, level - 1), level + 1)]} | {n_envs} envs | {a.minutes} min ===", flush=True)

    # ---------- 1-2: load yesterday's trainer, build today's simulation
    raw = make_raw_venv(rc, train_cfgs, n_envs, base_seed)
    if os.path.exists(S("trainer.zip")) and os.path.exists(S("trainer_vecnorm.pkl")):
        venv = VecNormalize.load(S("trainer_vecnorm.pkl"), raw)
        venv.training, venv.norm_reward = True, True
        model = PPO.load(S("trainer.zip"), env=venv, device="cpu")
        print(f"resumed trainer ({st['total_steps']:,} steps so far)", flush=True)
    else:
        venv = VecNormalize(raw, norm_obs=True, norm_reward=True, clip_obs=10.0)
        model = PPO("MlpPolicy", venv, n_steps=1024, batch_size=512, n_epochs=8, gamma=0.995, gae_lambda=0.95,
                    learning_rate=3e-4, ent_coef=0.003, policy_kwargs=dict(net_arch=[256, 256]), device="cpu", verbose=0)
        print("no checkpoint found: starting a fresh policy", flush=True)

    def save_trainer():
        model.save(S("trainer.zip"))
        venv.save(S("trainer_vecnorm.pkl"))

    # ---------- 3: PPO against reward points
    steps_before = model.num_timesteps
    cb = DailyCallback(time.time() + a.minutes * 60, S("training_log.csv"), save_trainer)
    model.learn(10 ** 10, callback=cb, reset_num_timesteps=False)
    save_trainer()
    trained = model.num_timesteps - steps_before
    st["total_steps"] += trained
    venv.close()

    # ---------- 4: fixed benchmark, challenger vs champion
    gate_cfg = LEVELS[level][1]
    print(f"evaluating challenger on {LEVELS[level][0]} ({a.eval_episodes} fixed episodes)...", flush=True)
    ch = evaluate(model, S("trainer_vecnorm.pkl"), rc, gate_cfg, a.eval_episodes)
    full = evaluate(model, S("trainer_vecnorm.pkl"), rc, LEVELS[2][1], max(5, a.eval_episodes // 2)) if level != 2 else ch
    champ_m = None
    if st["champion"] and os.path.exists(S("champion.zip")):
        champ_model = PPO.load(S("champion.zip"), device="cpu")
        champ_m = evaluate(champ_model, S("champion_vecnorm.pkl"), rc, gate_cfg, a.eval_episodes)

    # ---------- 5: promote only if better
    promoted = champ_m is None or score(ch) > score(champ_m) + 0.01
    if promoted:
        shutil.copy(S("trainer.zip"), S("champion.zip"))
        shutil.copy(S("trainer_vecnorm.pkl"), S("champion_vecnorm.pkl"))
        dim = export_policy_onnx(model, S("trainer_vecnorm.pkl"), S("policy.onnx"))
        st["champion"] = {"date": today, "level": LEVELS[level][0], "score": score(ch), "metrics": ch, "obs_dim": dim,
                          "total_steps": st["total_steps"]}
        st["bad_runs"] = 0
    else:
        st["bad_runs"] += int(score(ch) < score(champ_m) - 0.1)
        if st["bad_runs"] >= 2:      # trainer has drifted well below the champion: restart from champion
            shutil.copy(S("champion.zip"), S("trainer.zip"))
            shutil.copy(S("champion_vecnorm.pkl"), S("trainer_vecnorm.pkl"))
            st["bad_runs"] = 0
            print("trainer reset to champion after 2 weak runs", flush=True)

    # ---------- 6: curriculum
    leveled = promoted and ch["success"] >= a.promote_level_at and level < len(LEVELS) - 1
    if leveled:
        st["level"] += 1
    st["runs"] += 1
    st["history"].append({"date": today, "level": LEVELS[level][0], "steps_today": int(trained), "total_steps": st["total_steps"],
                          "challenger_score": round(score(ch), 4), "challenger_success": round(ch["success"], 3),
                          "champion_score": None if champ_m is None else round(score(champ_m), 4),
                          "promoted": bool(promoted), "leveled_up": bool(leveled)})
    st["history"] = st["history"][-90:]
    json.dump(st, open(S("state.json"), "w"), indent=1)
    json.dump({"date": today, "level": LEVELS[level][0], "challenger": ch, "full_difficulty": full,
               "champion_before": champ_m, "promoted": promoted}, open(S("metrics.json"), "w"), indent=1)

    # ---------- 7: report
    champ_txt = "none" if champ_m is None else "%.0f%% (score %.3f)" % (100 * champ_m["success"], score(champ_m))
    lines = [f"# FireBot daily training: {today}", "",
             f"Level **{LEVELS[level][0]}** -> next **{LEVELS[st['level']][0]}** | steps today **{trained:,}** | total **{st['total_steps']:,}**",
             f"Challenger success **{ch['success']:.0%}**, people {ch['people_rate']:.0%}, fires {ch['fire_rate']:.0%}, "
             f"collisions/ep {ch['collisions_per_ep']:.0f} (score {score(ch):.3f})",
             f"Champion before: {champ_txt} | "
             f"**{'PROMOTED' if promoted else 'not promoted'}**" + (" | level up" if leveled else ""),
             f"Full-difficulty (L2) success: {full['success']:.0%}", "",
             "## Reward points per episode (challenger)", "| term | points |", "|---|---|"]
    lines += [f"| {k} | {v:+.2f} |" for k, v in ch["reward_points_per_episode"].items()]
    lines += ["", "## History", "| date | level | steps | challenger | champion | promoted |", "|---|---|---|---|---|---|"]
    for h in st["history"][-14:]:
        lines.append(f"| {h['date']} | {h['level']} | {h['steps_today']:,} | {h['challenger_success']:.0%} | "
                     f"{h['champion_score'] if h['champion_score'] is not None else '-'} | {'yes' if h['promoted'] else 'no'} |")
    report = "\n".join(lines)
    open(S("report.md"), "w").write(report)
    print("\n" + report, flush=True)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        open(os.environ["GITHUB_STEP_SUMMARY"], "a").write(report + "\n")


if __name__ == "__main__":
    main()
