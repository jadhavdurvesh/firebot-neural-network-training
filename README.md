# FireBot training simulation + daily RL pipeline + on-robot brain

Physics simulation (PyBullet) where a car-like firefighting/rescue robot **learns by reinforcement learning**,
gets retrained **several times a day by GitHub Actions** in fresh randomised worlds, and is exported as an **ONNX "brain"**
for the real robot. Everything physical is **assumed** until hardware is final (`firebot_sim/config.py`).

## How the learning works (same idea as NVIDIA Isaac Lab)
| Isaac Lab concept | Here |
|---|---|
| Reward manager: weighted reward terms | `EnvConfig` weights; every step records **named terms** (`info["reward_terms"]`) |
| Many parallel simulated robots | `SubprocVecEnv`, one env per CPU core |
| Domain randomisation | mass, friction, motor torque, latency, sensor noise, smoke, battery, cooling faults, random maps |
| Curriculum | levels L0 easy -> L3 hard; moves up when champion success >= 80% |
| Train in sim, deploy checkpoint | PPO -> ONNX -> `firebot_brain` (robot-side code identical to sim-side) |
| Resume from checkpoint in a new scene | every run loads the previous model and trains in **new worlds** (seed = date + run counter) |

### The points the robot earns / loses
| Behaviour | Points |
|---|---|
| Find a person (first detection) | +1 |
| **Reach a person (located)** | **+20** |
| See a fire | +0.5 |
| **Extinguish a fire** (only counts after all people are located) | **+20** |
| **Mission complete** | **+50** |
| Move closer to the current known goal | +0.5 per metre |
| Explore a new area | +0.05 per new zone |
| Collision (per step touching something) | -0.5 |
| Spray water with no fire hit | -0.03 per step |
| Overheating risk | up to -0.02 per step |
| Time passing | -0.01 per step |
| Overheat / battery dead / rollover | -20, episode ends |

Change the numbers in `EnvConfig`; the daily report shows how many points came from each term, so you can spot
reward hacking (e.g. a policy that avoids crashes by standing still).

## What the robot senses, and the worlds it trains in
| Sensor (simulated) | What the policy gets | Honest status |
|---|---|---|
| 24-ray **lidar**, 6 m, noise + dropout | distance to walls/furniture/people all around | physically modelled (ray casting); smoke adds noise and dropouts |
| **IR strip**, 16 bins over 120 deg, 10 m | heat per bearing: tells it **which way the fire is**, even with no camera view; blocked by walls, **not** degraded by smoke | modelled from fire distance/intensity + line of sight; real sensor curves TBD |
| **Camera detector** (stand-in for firebot-vision) | position of people/fires that are in view (90 deg, 8 m, line of sight, 97% per step), remembered afterwards | **not real vision**: no images; replaced later by your detector + mapping |
| Pose, speed, steering | where it is and how fast | given with small noise; real robot needs SLAM/odometry |
| Battery, water, temperature, cooling | own status | synthetic models |

**Worlds:** `open` rooms with obstacles, and **buildings**: a hallway with 4-6 rooms on both sides, doorways, some
doors between neighbouring rooms, furniture, **1-4 people and 1-2 fires placed inside rooms**; the robot starts in the
hallway. Curriculum: L0 easy -> L1 -> L2 full -> L3 hard (open) -> **L4 building -> L5 full building**.
The fire-fighting order is still: find all people first, then use IR to home in on the fire and spray.

## Scheduled pipeline (`.github/workflows/daily-train.yml`, every 6 hours = 4 runs/day)
```
00/06/12/18 UTC  ->  download model-latest release  ->  PPO in today's new worlds (N minutes, all cores)
           ->  fixed benchmark: challenger vs champion  ->  better? promote + export policy.onnx
           ->  level up if success >= 80%  ->  upload to Releases (model-latest + dated)  ->  job summary report
```
Every run also publishes its own release `run-YYYY-MM-DD-HHMM` with a **3D video** (chase view + map + HUD, same benchmark
worlds every run so progress is visible over time) and **every parameter** (`run_record.json`, per-episode CSVs, training
curve): see **docs/RUN_RECORD.md**. `python scripts/make_report_figures.py` turns all runs into paper-ready figures and tables.
Setup: **docs/GITHUB_SETUP.md**. Get the model: `gh release download model-latest -p policy.onnx`.

## Local quick start
```bash
pip install -r requirements.txt
python -m pytest -q tests                                   # physics, sim<->robot parity, safety layer
python scripts/daily_train.py --state-dir state --minutes 10 --n-envs 4   # one scheduled run, locally
python scripts/run_brain_in_sim.py --onnx state/policy.onnx --episodes 10 # the real deployment code path, in sim
python scripts/record_demo.py --out demo.gif                # quick top-down GIF (scripted teacher)
python scripts/make_report_figures.py --state-dir state --out paper   # figures + tables from all runs
```

## On the robot
`firebot_brain/`: `SensorFrame` in -> safe `Command` out. Implement only `hardware_template.py` (lidar, pose, battery,
drivers). Memory, the 104-float observation (`firebot_sim/obs.py`), the ONNX policy and the safety supervisor are the code
that ran in simulation. Migration to real hardware: write measured values into a JSON like `configs/robot_assumed.json`,
retrain with `--robot-config`, widen randomisation to cover the remaining gap.
**Always** start with wheels lifted, e-stop in reach, speed cap lowered. The safety layer is not a certified function.

## Honest status
- Built and tested: simulator, reward-term logging, train/gate/promote/export pipeline (run end to end locally on
  short runs, resume + champion comparison verified), brain package, safety rules, workflows (YAML validated).
- **Buildings are hard.** A planner that sees the true map succeeds in 7/8 building episodes (so it is solvable), but my
  sensor-only scripted teacher (lidar + IR + memory) succeeds in 0/8 and finds ~50% of people: finding doors and exploring
  room by room needs more than reactive steering. Expect L4/L5 to take far longer than open levels. Next steps below.
- **Not yet proven: that PPO reaches a good policy.** In a 1-CPU sandbox, PPO from scratch scored ~0% after ~100k steps
  and imitation (DAgger) gave a weak student (0-17%). The scheduled runs on 2-4 cores over weeks are the real test;
  the scripted teacher (`firebot_sim/teacher.py`, 50-68% success) is the bar to beat. If curves stay flat, tune the
  rewards (watch the per-term report), add a recurrent policy, or bootstrap with `scripts/distill_then_ppo.py`.
- The GitHub workflows have not been executed on GitHub (no access from here): run the first one manually with 5-10 minutes.
- Sim limits: flat ground, rigid world, no fire spread, illustrative thermal/battery models.
