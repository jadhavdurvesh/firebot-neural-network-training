"""Turn the cumulative data from every run into paper-ready figures and tables.

  gh release download model-latest -p training_log_all.csv -p eval_episodes_all.csv -p runs_index.jsonl --dir state
  python scripts/make_report_figures.py --state-dir state --out paper

Outputs (PNG + PDF figures, Markdown + LaTeX tables) in --out:
  learning_curve.*   success / people / fires vs cumulative training steps (all runs)
  reward_points.*    mean points per reward term vs steps (shows what the robot is actually earning)
  eval_by_run.*      fixed-benchmark success per run with 95% confidence intervals
  runs_table.md/.tex one row per run (steps, level, success, collisions, promoted, commit, video)
"""
import argparse
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def save(fig, out, name):
    fig.tight_layout()
    fig.savefig(os.path.join(out, name + ".png"), dpi=200)
    fig.savefig(os.path.join(out, name + ".pdf"))
    plt.close(fig)


def smooth(y, k=5):
    y = np.asarray(y, float)
    if len(y) < k:
        return y
    return np.convolve(np.pad(y, (k // 2, k - 1 - k // 2), mode="edge"), np.ones(k) / k, mode="valid")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state-dir", default="state")
    ap.add_argument("--out", default="paper")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    made = []

    tl = os.path.join(a.state_dir, "training_log_all.csv")
    if os.path.exists(tl) and len(rows := read_csv(tl)) > 1:
        x = np.array([float(r["timesteps"]) for r in rows])
        fig, ax = plt.subplots(figsize=(7, 4))
        for key, lab in (("success", "mission success"), ("people", "people located"), ("fires", "fires extinguished")):
            ax.plot(x, smooth([float(r[key]) for r in rows]), label=lab)
        ax.set(xlabel="cumulative training steps", ylabel="rate (rolling 200 training episodes)", ylim=(0, 1.02),
               title="Learning curve across all runs")
        ax.legend(), ax.grid(alpha=.3)
        save(fig, a.out, "learning_curve"), made.append("learning_curve")

        terms = [k[4:] for k in rows[0] if k.startswith("pts_")]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for t in terms:
            y = smooth([float(r["pts_" + t]) for r in rows])
            if np.abs(y).max() > 0.05:
                ax.plot(x, y, label=t)
        ax.set(xlabel="cumulative training steps", ylabel="mean points per episode",
               title="Where the reward points come from")
        ax.legend(fontsize=7, ncol=2), ax.grid(alpha=.3)
        save(fig, a.out, "reward_points"), made.append("reward_points")

    idx = os.path.join(a.state_dir, "runs_index.jsonl")
    if os.path.exists(idx):
        runs = [json.loads(l) for l in open(idx) if l.strip()]
        ev = os.path.join(a.state_dir, "eval_episodes_all.csv")
        ci = {}
        if os.path.exists(ev):
            for r in read_csv(ev):
                if r["split"] == "challenger":
                    ci.setdefault(r["run"], []).append(r["end_reason"] == "success")
        if runs:
            xs = np.arange(len(runs))
            sr = np.array([r["success"] for r in runs])
            hw = np.array([1.96 * np.sqrt(max(p * (1 - p), 1e-9) / max(1, len(ci.get(r["tag"], [1]))))
                           for p, r in zip(sr, runs)])
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.errorbar(xs, sr, yerr=hw, fmt="o-", capsize=3)
            ax.set(xlabel="run number", ylabel="success on fixed benchmark seeds", ylim=(0, 1.02),
                   title="Fixed-benchmark success per run (95% CI)")
            ax.grid(alpha=.3)
            save(fig, a.out, "eval_by_run"), made.append("eval_by_run")
            head = "| run | level | steps (total) | success | people | fires | collisions/ep | promoted | commit | video |\n|---|---|---|---|---|---|---|---|---|---|\n"
            md = head + "\n".join(
                f"| {r['tag']} | {r['level']} | {r['total_steps']:,} | {r['success']:.0%} | {r['people_rate']:.0%} | "
                f"{r['fire_rate']:.0%} | {r['collisions_per_ep']:.0f} | {'yes' if r['promoted'] else 'no'} | "
                f"{(r.get('git_commit') or '')[:7]} | {r.get('video') or '-'} |" for r in runs)
            open(os.path.join(a.out, "runs_table.md"), "w").write(md + "\n")
            tex = ["\\begin{tabular}{llrrrrr}", "\\hline", "Run & Level & Steps & Success & People & Fires & Coll./ep \\\\", "\\hline"]
            for r in runs:
                tex.append(f"{r['tag'].replace('_', '\\_')} & {r['level'].replace('_', '\\_')} & {r['total_steps']:,} & "
                           f"{100*r['success']:.0f}\\% & {100*r['people_rate']:.0f}\\% & {100*r['fire_rate']:.0f}\\% & "
                           f"{r['collisions_per_ep']:.0f} \\\\".replace(",", "{,}"))
            tex += ["\\hline", "\\end{tabular}"]
            open(os.path.join(a.out, "runs_table.tex"), "w").write("\n".join(tex) + "\n")
            made += ["runs_table.md", "runs_table.tex"]
    print("wrote to", a.out, ":", made or "nothing (no data yet)")


if __name__ == "__main__":
    main()
