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

## Scheduled pipeline (`.github/workflows/daily-train.yml`, every 6 hours = 4 runs/day)
```
00/06/12/18 UTC  ->  download model-latest release  ->  PPO in today's new worlds (N minutes, all cores)
           ->  fixed benchmark: challenger vs champion  ->  better? promote + export policy.onnx
           ->  level up if success >= 80%  ->  upload to Releases (model-latest + dated)  ->  job summary report
```
Setup: **docs/GITHUB_SETUP.md**. Get the model: `gh release download model-latest -p policy.onnx`.

## Local quick start
```bash
pip install -r requirements.txt
python -m pytest -q tests                                   # physics, sim<->robot parity, safety layer
python scripts/daily_train.py --state-dir state --minutes 10 --n-envs 4   # one scheduled run, locally
python scripts/run_brain_in_sim.py --onnx state/policy.onnx --episodes 10 # the real deployment code path, in sim
python scripts/record_demo.py --out demo.gif                # top-down GIF (scripted teacher)
```

## On the robot
`firebot_brain/`: `SensorFrame` in -> safe `Command` out. Implement only `hardware_template.py` (lidar, pose, battery,
drivers). Memory, the 88-float observation (`firebot_sim/obs.py`), the ONNX policy and the safety supervisor are the code
that ran in simulation. Migration to real hardware: write measured values into a JSON like `configs/robot_assumed.json`,
retrain with `--robot-config`, widen randomisation to cover the remaining gap.
**Always** start with wheels lifted, e-stop in reach, speed cap lowered. The safety layer is not a certified function.

## Honest status
- Built and tested: simulator, reward-term logging, train/gate/promote/export pipeline (run end to end locally on
  short runs, resume + champion comparison verified), brain package, safety rules, workflows (YAML validated).
- **Not yet proven: that PPO reaches a good policy.** In a 1-CPU sandbox, PPO from scratch scored ~0% after ~100k steps
  and imitation (DAgger) gave a weak student (0-17%). The scheduled runs on 2-4 cores over weeks are the real test;
  the scripted teacher (`firebot_sim/teacher.py`, 50-68% success) is the bar to beat. If curves stay flat, tune the
  rewards (watch the per-term report), add a recurrent policy, or bootstrap with `scripts/distill_then_ppo.py`.
- The GitHub workflows have not been executed on GitHub (no access from here): run the first one manually with 5-10 minutes.
- Sim limits: flat ground, rigid world, no fire spread, illustrative thermal/battery models.
