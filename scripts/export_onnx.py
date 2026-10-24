"""Export a trained PPO policy to ONNX.  python scripts/export_onnx.py --model X.zip --vecnorm X_vecnorm.pkl"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stable_baselines3 import PPO
from firebot_sim.export import export_policy_onnx

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--vecnorm", default=None)
ap.add_argument("--out", default="firebot_policy.onnx")
a = ap.parse_args()
dim = export_policy_onnx(PPO.load(a.model, device="cpu"), a.vecnorm, a.out)
print("wrote", a.out, "| obs dim", dim, "| action = [throttle, steer, pump] in [-1, 1]")
