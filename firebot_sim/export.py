"""Policy -> ONNX export (observation normalisation baked in, so the robot feeds RAW observations)."""
from __future__ import annotations

import pickle

import numpy as np
import torch


def export_policy_onnx(model, vecnorm_path: str | None, out_path: str) -> int:
    mean = np.zeros(model.observation_space.shape, np.float32)
    std = np.ones_like(mean)
    clip = 10.0
    if vecnorm_path:
        with open(vecnorm_path, "rb") as f:
            vn = pickle.load(f)
        mean = vn.obs_rms.mean.astype(np.float32)
        std = np.sqrt(vn.obs_rms.var + vn.epsilon).astype(np.float32)
        clip = float(vn.clip_obs)

    class Wrapper(torch.nn.Module):
        def __init__(self, policy):
            super().__init__()
            self.policy = policy
            self.mean, self.std = torch.tensor(mean), torch.tensor(std)

        def forward(self, obs):
            x = torch.clamp((obs - self.mean) / self.std, -clip, clip)
            return torch.clamp(self.policy._predict(x, deterministic=True), -1.0, 1.0)

    policy = model.policy.to("cpu").eval()
    dummy = torch.zeros(1, *model.observation_space.shape)
    torch.onnx.export(Wrapper(policy).eval(), dummy, out_path, input_names=["obs"], output_names=["action"],
                      dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}}, opset_version=17, dynamo=False)
    return int(dummy.shape[1])
