"""3D video recorder: produces a playable file with HUD frames (headless CPU renderer)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from firebot_sim import FireBotEnv, WorldConfig
from firebot_sim.video import record_episodes


def test_records_a_video_with_metadata(tmp_path):
    env = FireBotEnv(world_cfg=WorldConfig(n_people=(1, 1), n_fires=(1, 1), n_obstacles=(0, 1), n_walls=(0, 0)))
    path = str(tmp_path / "t.mp4")
    meta = record_episodes(path, [("test", env, 3)], lambda obs: np.array([0.3, 0.0, -1.0], np.float32),
                           label="pytest", fps=10, every=2, max_steps=24, hold_s=0.5)
    env.close()
    assert os.path.getsize(meta["path"]) > 5_000
    assert meta["frames"] >= 12 and meta["episodes"][0]["steps"] == 24
    assert meta["encoder"] in ("mp4/h264", "gif")
