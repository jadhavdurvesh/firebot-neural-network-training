# Deploy to GitHub and run the scheduled training

## 1. Push the project
```bash
cd firebot-training-sim
git init -b main && git add . && git commit -m "FireBot training sim"
gh repo create jadhavdurvesh/firebot-training-sim --public --source=. --push     # or create it on github.com and git push
```
**Public repo (recommended, your plan):** standard GitHub-hosted runners are free and not metered for public repos,
so frequent runs cost nothing. Everything you publish (code, `policy.onnx`, reports) is public: keep secrets, private
datasets and proprietary hardware numbers out of the repo. (A *private* free account gets only ~2,000 min/month,
about 7 runs of 270 min.) No secrets are needed: the workflow uses the built-in `GITHUB_TOKEN`.
Check GitHub's current Actions terms and runner specs; this project assumes a ~4-core runner and 6 h job limit.

## 2. Allow the workflow to write releases
Repo -> Settings -> Actions -> General -> Workflow permissions -> **Read and write permissions** -> Save.
(The workflow already requests `contents: write`; this setting only matters if your org restricts it.)

## 3. First run (manual, short)
Actions -> **daily-train** -> Run workflow -> minutes = `10`. The first run builds PyBullet from source (~10 min),
cached afterwards. When it finishes, open the run's **Summary** page: that is the daily report (reward points per term,
challenger vs champion, history). A **Releases** entry `model-latest` will contain `policy.onnx`, `state.json`, checkpoints.

## 4. Let it run (several times a day)
The cron `0 */6 * * *` (UTC) starts automatically on the default branch: 4 runs/day. Each run: resume -> train in
new worlds -> gate against the champion -> promote -> publish to Releases.

| Want | Edit in `daily-train.yml` |
|---|---|
| 4 runs/day, ~4.5 h each (default) | `cron: "0 */6 * * *"`, `minutes` default `270` |
| 8 runs/day | `cron: "0 */3 * * *"`, `minutes` default `150` |
| 2 runs/day | `cron: "0 */12 * * *"`, `minutes` default `300` (job limit is 360 min, so a run cannot use the whole 12 h gap) |

Rule: **minutes + ~25 (install, evaluation) must be shorter than the gap between runs, and under 360.** If a run is
still going when the next fires, GitHub queues it (one pending run at a time), so nothing overlaps or corrupts the model.
More runs with the same total compute is not faster learning: what matters is total training time. Frequent shorter runs
mainly give you fresher models, fresher reports and more chances to catch a broken reward early.

Notes:
- GitHub disables scheduled workflows on public repos with **no repository activity for 60 days**; re-enable them in the
  Actions tab (or push any commit).
- Scheduled runs can start several minutes late. Release tags are `model-YYYY-MM-DD-HHMM` (one per promotion).
- To restart learning from scratch: Run workflow -> `fresh_start` = true.

## 5. Use the model
```bash
gh release download model-latest -p policy.onnx -p state.json -p metrics.json
python scripts/run_brain_in_sim.py --onnx policy.onnx --episodes 20      # sanity check in simulation first
```
Then deploy with `firebot_brain` (see README). Only trust a model whose `metrics.json` shows high success on the
fixed benchmark **and** that you have bench-tested with wheels lifted.

## 6. Tuning knobs
- `--minutes`, `--n-envs` (default: all cores), `--promote-level-at` (default 0.8), `--eval-episodes` (default 30).
- Rewards: `EnvConfig` in `firebot_sim/config.py`. Worlds/levels: `LEVELS` in `scripts/daily_train.py`.
- Real hardware numbers: `configs/robot_assumed.json` -> `--robot-config`.
