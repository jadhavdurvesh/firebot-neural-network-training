# What every run records (for reports and papers)

Every scheduled run publishes a GitHub Release tagged `run-YYYY-MM-DD-HHMM` containing:

| File | What it is |
|---|---|
| `run_video.mp4` | 3D chase-camera view + top-down map with a HUD (points so far, people/fires, battery, temperature, collisions). The same **fixed benchmark worlds (same seeds) every run**, so videos from different weeks are directly comparable. Plays at 2.25x real time. |
| `run_record.json` | Every parameter of the run (schema below) |
| `eval_episodes.csv` | One row per benchmark episode: seed, end reason, steps, return, people/fires, collisions, battery, max temperature, and the points earned from each reward term |
| `training_log.csv` | Training curve for this run: success/people/fires/collisions/episode length and mean points per reward term, every ~20k steps |
| `metrics.json`, `report.md` | Summary numbers and the human-readable report |
| `policy.onnx` | The exported model (only in runs where the challenger was promoted to champion) |

The `model-latest` release additionally holds **cumulative data from all runs**: `training_log_all.csv`,
`eval_episodes_all.csv`, `runs_index.jsonl` (one line per run), plus the checkpoints that carry training forward.

## `run_record.json` (schema `firebot-run-record/1`)
- `run`: tag, run number, UTC start/end, wall-clock and training seconds, minutes budget, **world seed**, resumed or fresh
- `ci`: repository, workflow, GitHub run id/URL, event (schedule/manual), runner OS
- `code`: git commit SHA, dirty flag, ref  (**cite this commit to reproduce a result**)
- `software`: Python, OS, CPU count, numpy, torch, stable-baselines3, gymnasium, pybullet API, onnxruntime, imageio, Pillow
- `training`: all PPO hyperparameters, network architecture, parallel envs, steps this run and total, steps/second, curriculum level and train worlds, level-up
- `simulation`: full `RobotConfig` (all **assumed** hardware numbers), full `EnvConfig` (**all reward weights**, physics/control rate, limits),
  every curriculum level's world settings, the **domain-randomisation table**, observation size, action definition
- `evaluation`: benchmark seeds, episodes, score formula, challenger / full-difficulty / previous-champion metrics (with 95% CI on success), promotion decision
- `model`: SHA-256 of trainer, champion and ONNX files
- `video`: encoder, fps, frames, duration, size, playback speed, and the outcome of each recorded episode
- `files_in_this_run_with_sha256`: integrity hashes of the other run files

## Making figures and tables
```bash
gh release download model-latest -p training_log_all.csv -p eval_episodes_all.csv -p runs_index.jsonl --dir state
python scripts/make_report_figures.py --state-dir state --out paper
```
Produces `learning_curve`, `reward_points`, `eval_by_run` (PNG + PDF) and `runs_table.md` / `runs_table.tex`.
Download any run's video and record: `gh release download run-2026-11-02-0600`.

## Honest notes for a paper
- The robot, motors, sensors, battery and thermal numbers are **assumptions** until hardware is final; say so, and cite the commit.
- Success is measured on a fixed seed set with a deterministic policy; report the confidence interval, not just the mean.
- Training curves use stochastic training episodes, so they sit below the deterministic benchmark numbers.
- Videos show the policy on benchmark worlds, not a hardware experiment.
- Disk/retention: run releases accumulate (~1-5 MB each; 4 runs/day = ~1,500/year). GitHub does not bill public release assets,
  but you can prune old `run-*` releases with `gh release delete` if you want a tidy page; keep the ones you cite.
