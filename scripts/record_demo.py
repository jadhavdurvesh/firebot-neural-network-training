"""Render an episode (oracle or model) to a GIF for presentations. Frames are top-down, simulated."""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from PIL import Image
from firebot_sim import FireBotEnv
from firebot_sim.baseline import OraclePlanner

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="demo.gif")
ap.add_argument("--seed", type=int, default=3)
ap.add_argument("--every", type=int, default=4)
ap.add_argument("--max-frames", type=int, default=150)
a = ap.parse_args()
env = FireBotEnv(render_mode="rgb_array")
obs, _ = env.reset(seed=a.seed)
planner, frames, done, t = OraclePlanner(env), [], False, 0
while not done and len(frames) < a.max_frames:
    obs, r, term, trunc, info = env.step(planner.act())
    done = term or trunc
    if t % a.every == 0:
        frames.append(Image.fromarray(env.render()).resize((320, 320)))
    t += 1
frames[0].save(a.out, save_all=True, append_images=frames[1:], duration=80, loop=0)
print("wrote", a.out, len(frames), "frames;", info["end_reason"])
