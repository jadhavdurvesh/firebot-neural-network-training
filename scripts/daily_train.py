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
import hashlib
import pickle
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from firebot_sim import EnvConfig, FireBotEnv, RobotConfig, WorldConfig
from firebot_sim.config import DOMAIN_RANDOMIZATION
from firebot_sim.export import export_policy_onnx

LEVELS = [
    ("L0_easy", WorldConfig(n_people=(1, 1), n_fires=(1, 1), n_obstacles=(0, 2), n_walls=(0, 0))),
    ("L1_medium", WorldConfig(n_people=(1, 1), n_fires=(1, 1), n_obstacles=(2, 5), n_walls=(0, 1))),
    ("L2_full", WorldConfig()),
    ("L3_hard", WorldConfig(n_people=(2, 3), n_fires=(1, 2), n_obstacles=(5, 9), n_walls=(1, 3))),
    # multi-room buildings: hallway + 4-6 rooms, doors, furniture, people and fires inside rooms
    ("L4_building", WorldConfig(layout="building", n_people=(1, 2), n_fires=(1, 1), n_rooms_per_side=(2, 2))),
    ("L5_building_full", WorldConfig(layout="building", n_people=(2, 4), n_fires=(1, 2))),
]
EVAL_SEED0 = 100_000          # fixed benchmark seeds: identical every day


REWARD_TERMS = ["time_penalty", "collision", "detect_person", "detect_fire", "person_located", "fire_extinguished",
                "wasted_pump", "thermal_risk", "exploration", "progress_to_goal", "mission_success",
                "fail_thermal", "fail_battery", "fail_rollover"]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_info():
    def run(*args):
        try:
            return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception:
            return ""
    sha = run("rev-parse", "HEAD") or os.environ.get("GITHUB_SHA", "")
    return {"commit": sha, "dirty": bool(run("status", "--porcelain")), "ref": os.environ.get("GITHUB_REF", "")}


def software_versions():
    v = {"python": platform.python_version(), "platform": platform.platform(), "machine": platform.machine(),
         "cpu_count": os.cpu_count(), "numpy": np.__version__}
    for mod in ("torch", "stable_baselines3", "gymnasium", "onnxruntime", "imageio", "PIL"):
        try:
            v[mod] = __import__(mod).__version__
        except Exception:
            v[mod] = None
    try:
        import pybullet
        v["pybullet_api"] = pybullet.getAPIVersion()
    except Exception:
        v["pybullet_api"] = None
    return v


def ci_info():
    repo, run = os.environ.get("GITHUB_REPOSITORY"), os.environ.get("GITHUB_RUN_ID")
    server = os.environ.get("GITHUB_SERVER_URL", "https://github.com")
    return {"running_in_github_actions": bool(os.environ.get("GITHUB_ACTIONS")), "repository": repo,
            "workflow": os.environ.get("GITHUB_WORKFLOW"), "run_id": run, "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
            "event": os.environ.get("GITHUB_EVENT_NAME"), "runner_os": os.environ.get("RUNNER_OS"),
            "run_url": f"{server}/{repo}/actions/runs/{run}" if repo and run else None}


def ppo_hparams(model):
    lr = model.learning_rate
    return {"algorithm": "PPO (stable-baselines3)", "learning_rate": lr if isinstance(lr, (int, float)) else "schedule",
            "n_steps_per_env": model.n_steps, "batch_size": model.batch_size, "n_epochs": model.n_epochs,
            "gamma": model.gamma, "gae_lambda": model.gae_lambda, "ent_coef": model.ent_coef, "vf_coef": model.vf_coef,
            "clip_range": model.clip_range(1.0) if callable(model.clip_range) else model.clip_range,
            "max_grad_norm": model.max_grad_norm, "net_arch": getattr(model, "policy_kwargs", {}).get("net_arch"),
            "observation_normalisation": True, "reward_normalisation": True, "device": str(model.device)}


def append_csv(path, header, rows):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        w.writerows(rows)


def make_act_fn(model, vecnorm_path):
    vn = pickle.load(open(vecnorm_path, "rb"))
    return lambda obs: model.predict(vn.normalize_obs(obs), deterministic=True)[0]


def score(m):
    """Single number used to decide promotion (higher is better)."""
    return m["success"] + 0.3 * m["people_rate"] + 0.3 * m["fire_rate"] - 0.0005 * m["collisions_per_ep"]


def evaluate(model, vecnorm_path, rc, wc, episodes, split, level_name):
    """Deterministic rollouts on FIXED seeds. Returns (aggregate metrics, per-episode rows)."""
    vn = pickle.load(open(vecnorm_path, "rb")) if vecnorm_path and os.path.exists(vecnorm_path) else None
    env = FireBotEnv(rc, wc)
    succ, ppl, fir, coll, steps, ret, reasons, terms, rows = [], [], [], [], [], [], {}, {}, []
    for k in range(episodes):
        seed = EVAL_SEED0 + k
        obs, _ = env.reset(seed=seed)
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
        rows.append([split, level_name, seed, info["end_reason"], info["steps"], round(total, 3),
                     info["people_located"], info["n_people"], info["fires_out"], info["n_fires"], info["collisions"],
                     round(info["battery"], 4), round(info["max_temp_c"], 2)] +
                    [round(info["reward_terms"].get(t, 0.0), 3) for t in REWARD_TERMS])
    env.close()
    m = {"success": float(np.mean(succ)), "people_rate": float(np.mean(ppl)), "fire_rate": float(np.mean(fir)),
         "collisions_per_ep": float(np.mean(coll)), "mean_steps": float(np.mean(steps)),
         "mean_return": float(np.mean(ret)), "episodes": episodes, "seeds": [EVAL_SEED0, EVAL_SEED0 + episodes - 1],
         "success_ci95_halfwidth": float(1.96 * np.sqrt(max(np.mean(succ) * (1 - np.mean(succ)), 1e-9) / episodes)),
         "end_reasons": reasons,
         "reward_points_per_episode": {k: round(v, 2) for k, v in sorted(terms.items(), key=lambda kv: -abs(kv[1]))}}
    return m, rows


EVAL_HEADER = ["split", "level", "seed", "end_reason", "steps", "return", "people_located", "n_people", "fires_out",
               "n_fires", "collisions", "battery_end", "max_temp_c"] + ["pts_" + t for t in REWARD_TERMS]


class DailyCallback(BaseCallback):
    HEADER = (["unix_time", "timesteps", "success", "people", "fires", "collisions_per_ep", "mean_ep_steps", "episodes"]
              + ["pts_" + t for t in REWARD_TERMS])

    def __init__(self, deadline, csv_path, ckpt_fn, every=20_000, ckpt_every_s=900):
        super().__init__()
        self.deadline, self.every, self.ckpt_fn, self.ckpt_every_s = deadline, every, ckpt_fn, ckpt_every_s
        self.t0 = self.t_ckpt = time.time()
        self.last = 0
        self.ep = deque(maxlen=200)
        self.f = open(csv_path, "a", newline="")
        self.w = csv.writer(self.f)
        if self.f.tell() == 0:
            self.w.writerow(self.HEADER)

    def _on_step(self):
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if done:
                self.ep.append([info["end_reason"] == "success", info["people_located"] / max(1, info["n_people"]),
                                info["fires_out"] / max(1, info["n_fires"]), info["collisions"], info["steps"]]
                               + [info["reward_terms"].get(t, 0.0) for t in REWARD_TERMS])
        if self.num_timesteps - self.last >= self.every and len(self.ep) >= 20:
            self.last = self.num_timesteps
            a = np.array(self.ep, float)
            print(f"[train] t={self.num_timesteps:>9} {time.time()-self.t0:6.0f}s success {a[:,0].mean():.2f} "
                  f"people {a[:,1].mean():.2f} fires {a[:,2].mean():.2f} coll/ep {a[:,3].mean():.0f}", flush=True)
            m = a.mean(0).round(4)
            self.w.writerow([int(time.time()), self.num_timesteps, *m[:5], len(a), *m[5:]])
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
    ap.add_argument("--state-dir", default="state", help="persistent carry-over between runs (model, state, indexes)")
    ap.add_argument("--out-dir", default="out", help="this run's deliverables (video, record, csv, report)")
    ap.add_argument("--minutes", type=float, default=120)
    ap.add_argument("--n-envs", type=int, default=0, help="0 = number of CPU cores")
    ap.add_argument("--eval-episodes", type=int, default=30)
    ap.add_argument("--video-episodes", type=int, default=3, help="0 disables the 3D video")
    ap.add_argument("--video-max-steps", type=int, default=1200)
    ap.add_argument("--date", default=None, help="YYYY-MM-DD; makes the run reproducible/labelled (tests)")
    ap.add_argument("--promote-level-at", type=float, default=0.8)
    ap.add_argument("--robot-config", default=None)
    a = ap.parse_args()

    t_wall0 = time.time()
    os.makedirs(a.state_dir, exist_ok=True)
    shutil.rmtree(a.out_dir, ignore_errors=True)
    os.makedirs(a.out_dir, exist_ok=True)
    S = lambda n: os.path.join(a.state_dir, n)
    O = lambda n: os.path.join(a.out_dir, n)
    st = load_state(S("state.json"))
    utc_start = dt.datetime.now(dt.timezone.utc)
    if a.date:                                       # tests / reproducible runs
        day, today = a.date, f"{a.date} #{st['runs'] + 1}"
        run_tag = f"run-{a.date}-n{st['runs'] + 1}"
    else:
        day, today = utc_start.date().isoformat(), utc_start.strftime("%Y-%m-%d %H:%MZ")
        run_tag = utc_start.strftime("run-%Y-%m-%d-%H%M")
    # new worlds EVERY run (not just every day): seed mixes the date with the run counter
    base_seed = (int(day.replace("-", "")) * 1000 + st["runs"]) % (2 ** 31 - 1)
    n_envs = a.n_envs or os.cpu_count() or 1
    rc = RobotConfig.from_json(a.robot_config) if a.robot_config else RobotConfig()
    level = st["level"]
    train_names = [LEVELS[i][0] for i in range(max(0, level - 1), level + 1)]
    train_cfgs = [LEVELS[i][1] for i in range(max(0, level - 1), level + 1)]
    print(f"=== {today} | seed {base_seed} | level {LEVELS[level][0]} | train worlds: {train_names} | "
          f"{n_envs} envs | {a.minutes} min ===", flush=True)

    # ---------- 1-2: load the previous trainer, build this run's simulation
    raw = make_raw_venv(rc, train_cfgs, n_envs, base_seed)
    resumed = os.path.exists(S("trainer.zip")) and os.path.exists(S("trainer_vecnorm.pkl"))
    steps_at_start = st["total_steps"]
    if resumed:
        venv = VecNormalize.load(S("trainer_vecnorm.pkl"), raw)
        venv.training, venv.norm_reward = True, True
        model = PPO.load(S("trainer.zip"), env=venv, device="cpu")
        print(f"resumed trainer ({st['total_steps']:,} steps so far)", flush=True)
    else:
        venv = VecNormalize(raw, norm_obs=True, norm_reward=True, clip_obs=10.0)
        model = PPO("MlpPolicy", venv, n_steps=1024, batch_size=512, n_epochs=8, gamma=0.995, gae_lambda=0.95,
                    learning_rate=3e-4, ent_coef=0.003, policy_kwargs=dict(net_arch=[256, 256]), device="cpu", verbose=0)
        print("no checkpoint found: starting a fresh policy", flush=True)
    hparams = ppo_hparams(model)

    def save_trainer():
        model.save(S("trainer.zip"))
        venv.save(S("trainer_vecnorm.pkl"))

    # ---------- 3: PPO against reward points
    steps_before = model.num_timesteps
    t_train0 = time.time()
    cb = DailyCallback(time.time() + a.minutes * 60, O("training_log.csv"), save_trainer)
    model.learn(10 ** 10, callback=cb, reset_num_timesteps=False)
    train_seconds = time.time() - t_train0
    save_trainer()
    trained = model.num_timesteps - steps_before
    st["total_steps"] += trained
    venv.close()

    # ---------- 4: fixed benchmark, challenger vs champion
    gate_name, gate_cfg = LEVELS[level]
    print(f"evaluating challenger on {gate_name} ({a.eval_episodes} fixed episodes)...", flush=True)
    ch, rows_ch = evaluate(model, S("trainer_vecnorm.pkl"), rc, gate_cfg, a.eval_episodes, "challenger", gate_name)
    eval_rows = list(rows_ch)
    if level != 2:
        full, rows_full = evaluate(model, S("trainer_vecnorm.pkl"), rc, LEVELS[2][1], max(5, a.eval_episodes // 2),
                                   "challenger_full_difficulty", LEVELS[2][0])
        eval_rows += rows_full
    else:
        full = ch
    champ_m = None
    if st["champion"] and os.path.exists(S("champion.zip")):
        champ_model = PPO.load(S("champion.zip"), device="cpu")
        champ_m, rows_champ = evaluate(champ_model, S("champion_vecnorm.pkl"), rc, gate_cfg, a.eval_episodes,
                                       "champion_before", gate_name)
        eval_rows += rows_champ

    # ---------- 5: promote only if better
    promoted = champ_m is None or score(ch) > score(champ_m) + 0.01
    if promoted:
        shutil.copy(S("trainer.zip"), S("champion.zip"))
        shutil.copy(S("trainer_vecnorm.pkl"), S("champion_vecnorm.pkl"))
        dim = export_policy_onnx(model, S("trainer_vecnorm.pkl"), S("policy.onnx"))
        shutil.copy(S("policy.onnx"), O("policy.onnx"))
        st["champion"] = {"date": today, "level": gate_name, "score": score(ch), "metrics": ch, "obs_dim": dim,
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

    # ---------- 6b: 3D video of this run's policy on FIXED benchmark worlds (same seeds every run -> comparable)
    video = {"enabled": a.video_episodes > 0}
    if a.video_episodes > 0:
        try:
            from firebot_sim.video import record_episodes
            plan = [(f"challenger | {gate_name}", FireBotEnv(rc, gate_cfg), EVAL_SEED0),
                    (f"challenger | {gate_name}", FireBotEnv(rc, gate_cfg), EVAL_SEED0 + 1),
                    (f"challenger | {LEVELS[4][0]}", FireBotEnv(rc, LEVELS[4][1]), EVAL_SEED0)][:a.video_episodes]
            t_v = time.time()
            video.update(record_episodes(O("run_video.mp4"), plan, make_act_fn(model, S("trainer_vecnorm.pkl")),
                                         label=today, fps=15, every=3, max_steps=a.video_max_steps))
            video["playback_speed_vs_realtime"] = round(15 * 3 / 20, 2)
            video["render_seconds"] = round(time.time() - t_v, 1)
            video["path"] = os.path.basename(video["path"])
            for _, e, _ in plan:
                e.close()
            print(f"video: {video['path']} {video['duration_s']}s {video['size_bytes']/1e6:.1f} MB", flush=True)
        except Exception as exc:                   # never lose a training run because rendering failed
            video["error"] = f"{type(exc).__name__}: {exc}"
            print("VIDEO FAILED:", video["error"], flush=True)

    # ---------- 7: records
    st["runs"] += 1
    utc_end = dt.datetime.now(dt.timezone.utc)
    st["history"].append({"date": today, "level": gate_name, "steps_today": int(trained), "total_steps": st["total_steps"],
                          "challenger_score": round(score(ch), 4), "challenger_success": round(ch["success"], 3),
                          "champion_score": None if champ_m is None else round(score(champ_m), 4),
                          "promoted": bool(promoted), "leveled_up": bool(leveled)})
    st["history"] = st["history"][-90:]
    json.dump(st, open(S("state.json"), "w"), indent=1)
    json.dump({"date": today, "level": gate_name, "challenger": ch, "full_difficulty": full,
               "champion_before": champ_m, "promoted": promoted}, open(S("metrics.json"), "w"), indent=1)
    shutil.copy(S("metrics.json"), O("metrics.json"))

    # per-run CSVs (this run only) ...
    append_csv(O("eval_episodes.csv"), EVAL_HEADER, eval_rows)
    # ... and cumulative CSVs carried in the state dir (every run, ready for plots/tables in a paper)
    log_rows = list(csv.reader(open(O("training_log.csv"))))[1:]
    append_csv(S("training_log_all.csv"), ["run"] + DailyCallback.HEADER, [[run_tag] + r for r in log_rows])
    append_csv(S("eval_episodes_all.csv"), ["run"] + EVAL_HEADER, [[run_tag] + r for r in eval_rows])

    files = {n: sha256(O(n)) for n in os.listdir(a.out_dir)}
    record = {
        "schema": "firebot-run-record/1",
        "run": {"tag": run_tag, "label": today, "run_number": st["runs"], "utc_start": utc_start.isoformat(),
                "utc_end": utc_end.isoformat(), "wall_clock_s": round(time.time() - t_wall0, 1),
                "train_seconds": round(train_seconds, 1), "minutes_budget": a.minutes, "world_seed": base_seed,
                "resumed_from_previous_run": resumed},
        "ci": ci_info(), "code": git_info(), "software": software_versions(),
        "training": {"hyperparameters": hparams, "n_envs": n_envs, "steps_this_run": int(trained),
                     "steps_per_second": round(trained / max(train_seconds, 1e-9), 1),
                     "total_steps_before": steps_at_start, "total_steps_after": st["total_steps"],
                     "curriculum_level": gate_name, "train_worlds": train_names,
                     "next_level": LEVELS[st["level"]][0], "leveled_up": bool(leveled)},
        "simulation": {"robot_config_assumed_until_hardware_final": asdict(RobotConfig()) if not a.robot_config else asdict(rc),
                       "env_config_including_reward_weights": asdict(EnvConfig()),
                       "world_levels": {n: asdict(c) for n, c in LEVELS},
                       "domain_randomization": DOMAIN_RANDOMIZATION,
                       "observation": {"dim": int(model.observation_space.shape[0]), "layout": "see firebot_sim/obs.py"},
                       "action": "[throttle, steer, pump] in [-1, 1]"},
        "evaluation": {"benchmark_seeds_start": EVAL_SEED0, "episodes_gate": a.eval_episodes,
                       "score_formula": "success + 0.3*people_rate + 0.3*fire_rate - 0.0005*collisions_per_ep",
                       "challenger": ch, "challenger_full_difficulty": full, "champion_before": champ_m,
                       "challenger_score": score(ch), "champion_score": None if champ_m is None else score(champ_m),
                       "promoted": bool(promoted)},
        "model": {"trainer_zip_sha256": sha256(S("trainer.zip")), "champion_zip_sha256": sha256(S("champion.zip")),
                  "policy_onnx_sha256": sha256(S("policy.onnx")) if os.path.exists(S("policy.onnx")) else None},
        "video": video,
        "files_in_this_run_with_sha256": files,
    }
    json.dump(record, open(O("run_record.json"), "w"), indent=1, default=_jsonable)
    with open(S("runs_index.jsonl"), "a") as f:
        f.write(json.dumps({"tag": run_tag, "label": today, "utc_end": utc_end.isoformat(), "level": gate_name,
                            "steps_this_run": int(trained), "total_steps": st["total_steps"],
                            "success": ch["success"], "people_rate": ch["people_rate"], "fire_rate": ch["fire_rate"],
                            "collisions_per_ep": ch["collisions_per_ep"], "score": score(ch), "promoted": bool(promoted),
                            "git_commit": record["code"]["commit"], "run_url": record["ci"]["run_url"],
                            "video": video.get("path"), "video_error": video.get("error")}, default=_jsonable) + "\n")

    # ---------- 8: human-readable report
    champ_txt = "none" if champ_m is None else "%.0f%% (score %.3f)" % (100 * champ_m["success"], score(champ_m))
    repo = os.environ.get("GITHUB_REPOSITORY")
    rel = f"https://github.com/{repo}/releases/tag/{run_tag}" if repo else run_tag
    lines = [f"# FireBot training run: {today}", "",
             f"Level **{gate_name}** -> next **{LEVELS[st['level']][0]}** | steps this run **{trained:,}** "
             f"({trained / max(train_seconds, 1e-9):,.0f}/s on {n_envs} envs) | total **{st['total_steps']:,}**",
             f"Challenger success **{ch['success']:.0%}** (+/-{ch['success_ci95_halfwidth']:.0%} 95% CI), people {ch['people_rate']:.0%}, "
             f"fires {ch['fire_rate']:.0%}, collisions/ep {ch['collisions_per_ep']:.0f} (score {score(ch):.3f})",
             f"Champion before: {champ_txt} | **{'PROMOTED' if promoted else 'not promoted'}**" + (" | level up" if leveled else ""),
             f"Full-difficulty (L2) success: {full['success']:.0%}", "",
             f"Run files (3D video, parameters, per-episode CSVs): {rel}",
             "Video: " + (video.get("path", "FAILED: " + video.get("error", "disabled")) if video.get("enabled") else "disabled"), "",
             "## Reward points per episode (challenger)", "| term | points |", "|---|---|"]
    lines += [f"| {k} | {v:+.2f} |" for k, v in ch["reward_points_per_episode"].items()]
    lines += ["", "## History", "| run | level | steps | challenger | champion | promoted |", "|---|---|---|---|---|---|"]
    for h in st["history"][-14:]:
        lines.append(f"| {h['date']} | {h['level']} | {h['steps_today']:,} | {h['challenger_success']:.0%} | "
                     f"{h['champion_score'] if h['champion_score'] is not None else '-'} | {'yes' if h['promoted'] else 'no'} |")
    report = "\n".join(lines)
    open(S("report.md"), "w").write(report)
    open(O("report.md"), "w").write(report)
    open(O("release_notes.md"), "w").write(report)
    open(O("run_tag.txt"), "w").write(run_tag)
    print("\n" + report, flush=True)
    print("\nrun files:", {n: f"{os.path.getsize(O(n))/1e3:.0f} kB" for n in sorted(os.listdir(a.out_dir))}, flush=True)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        open(os.environ["GITHUB_STEP_SUMMARY"], "a").write(report + "\n")


if __name__ == "__main__":
    main()
