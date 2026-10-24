"""Sim-in-the-loop test of the exact on-robot code path: FireBotBrain (ONNX + memory + safety) driving the physics sim.

  python scripts/run_brain_in_sim.py --onnx firebot_policy.onnx --episodes 10 [--no-safety]
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from firebot_sim import FireBotEnv
from firebot_brain import FireBotBrain
from firebot_brain.sim_hal import SimHAL

ap = argparse.ArgumentParser()
ap.add_argument("--onnx", required=True)
ap.add_argument("--episodes", type=int, default=10)
ap.add_argument("--no-safety", action="store_true")
a = ap.parse_args()
env = FireBotEnv(); hal = SimHAL(env)
ok, lat, trips, reasons = 0, [], 0, {}
for ep in range(a.episodes):
    hal.reset(seed=7000 + ep)
    brain = FireBotBrain(a.onnx, n_people=env.n_people, n_fires=env.n_fires, safety=not a.no_safety)
    done = False
    while not done:
        cmd, info = brain.step(hal.read())
        lat.append(info["latency_ms"]); trips += bool(info["safety"])
        _, _, te, tr, einfo = hal.send(cmd); done = te or tr
    ok += einfo["end_reason"] == "success"; reasons[einfo["end_reason"]] = reasons.get(einfo["end_reason"], 0) + 1
    print(f"ep {ep} {einfo['end_reason']:<16} people {einfo['people_located']}/{einfo['n_people']} fires {einfo['fires_out']}/{einfo['n_fires']}", flush=True)
print(f"\nsuccess {ok}/{a.episodes} | policy+memory+safety latency mean {np.mean(lat):.2f} ms, p99 {np.percentile(lat,99):.2f} ms "
      f"| safety interventions {trips} | {reasons}")
