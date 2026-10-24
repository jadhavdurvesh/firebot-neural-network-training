"""Record 3D videos of a policy running in the physics simulation (headless, CPU renderer).

Each frame = chase-camera 3D view (left) + top-down map (right) + a HUD with the numbers a paper needs
(step, reward points so far, people/fires, battery, temperature, collisions). Output is MP4 (H.264, via the ffmpeg
bundled in `imageio-ffmpeg`) and falls back to an animated GIF if no encoder is available.
"""
from __future__ import annotations

import math
import os
from typing import Callable, List, Sequence, Tuple

import numpy as np
import pybullet as pb
from PIL import Image, ImageDraw, ImageFont

CHASE = (640, 360)
TOP = (360, 360)


def _font(size=15):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:                      # old Pillow
        return ImageFont.load_default()


def _grab(bc, view, proj, size):
    _, _, rgb, _, _ = bc.getCameraImage(size[0], size[1], view, proj, renderer=pb.ER_TINY_RENDERER, shadow=0)
    return np.reshape(rgb, (size[1], size[0], 4))[:, :, :3].astype(np.uint8)


def render_frame(env) -> np.ndarray:
    """Chase-camera 3D view + top-down view, side by side (height 360, width 1000)."""
    bc = env.bc
    pos, yaw, _ = env.robot.pose()
    c, s = math.cos(yaw), math.sin(yaw)
    eye = [pos[0] - 1.9 * c, pos[1] - 1.9 * s, 2.7]      # high enough to look over 1.2 m walls into rooms
    tgt = [pos[0] + 1.2 * c, pos[1] + 1.2 * s, 0.1]
    chase = _grab(bc, bc.computeViewMatrix(eye, tgt, [0, 0, 1]),
                  bc.computeProjectionMatrixFOV(65, CHASE[0] / CHASE[1], 0.1, 60), CHASE)
    A = env.wcfg.arena_half
    top = _grab(bc, bc.computeViewMatrix([0, 0, 2.3 * A], [0, 0, 0], [0, 1, 0]),
                bc.computeProjectionMatrixFOV(45, 1.0, 0.1, 60), TOP)
    return np.concatenate([chase, top], axis=1)


def add_hud(frame: np.ndarray, lines: Sequence[str], banner: str | None = None) -> np.ndarray:
    img = Image.fromarray(frame).convert("RGBA")
    over = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(over)
    f = _font(15)
    h = 6 + 19 * len(lines)
    d.rectangle([0, 0, 520, h], fill=(0, 0, 0, 150))
    for i, t in enumerate(lines):
        d.text((8, 4 + 19 * i), t, font=f, fill=(255, 255, 255, 255))
    if banner:
        fb = _font(30)
        w = d.textlength(banner, font=fb)
        d.rectangle([img.width // 2 - w / 2 - 14, img.height - 62, img.width // 2 + w / 2 + 14, img.height - 14],
                    fill=(0, 0, 0, 190))
        d.text((img.width // 2 - w / 2, img.height - 58), banner, font=fb, fill=(255, 220, 60, 255))
    return np.asarray(Image.alpha_composite(img, over).convert("RGB"))


class _Writer:
    def __init__(self, path, fps):
        self.fps, self.path, self.frames, self.encoder = fps, path, [], "mp4/h264"
        self.w = None
        if path.endswith(".mp4"):
            try:
                import imageio.v2 as imageio
                self.w = imageio.get_writer(path, fps=fps, codec="libx264", pixelformat="yuv420p",
                                            macro_block_size=2, quality=7)
            except Exception:
                self.w = None
        if self.w is None:                                     # GIF fallback
            self.path = os.path.splitext(path)[0] + ".gif"
            self.encoder = "gif"

    def add(self, frame):
        if self.w is not None:
            self.w.append_data(frame)
        else:
            self.frames.append(Image.fromarray(frame).resize((frame.shape[1] // 2, frame.shape[0] // 2)))

    def close(self):
        if self.w is not None:
            self.w.close()
        elif self.frames:
            self.frames[0].save(self.path, save_all=True, append_images=self.frames[1:],
                                duration=int(1000 / self.fps), loop=0)
        return self.path


def record_episodes(path: str, episodes: List[Tuple[str, object, int]], act_fn: Callable, label: str = "",
                    fps: int = 10, every: int = 2, max_steps: int = 1200, hold_s: float = 2.0) -> dict:
    """episodes: [(title, env, seed), ...]; act_fn(obs) -> action. Returns metadata for the run record."""
    wr = _Writer(path, fps)
    meta = {"episodes": [], "fps": fps, "render_every_n_control_steps": every, "view": "chase 3D + top-down"}
    n_frames = 0
    for k, (title, env, seed) in enumerate(episodes):
        obs, info = env.reset(seed=seed)
        total, t, done = 0.0, 0, False
        banner = None
        while not done and t < max_steps:
            obs, r, te, tr, info = env.step(act_fn(obs))
            total += r
            t += 1
            done = te or tr
            if t % every == 0 or done:
                hud = [f"{label}   {title}   seed {seed}   step {t}",
                       f"points {total:+8.1f}   people {info['people_located']}/{info['n_people']}   "
                       f"fires out {info['fires_out']}/{info['n_fires']}",
                       f"battery {100 * info['battery']:.0f}%   temp {info['temp_c']:.0f}C   "
                       f"water {100 * info['water']:.0f}%   collisions {info['collisions']}"]
                wr.add(add_hud(render_frame(env), hud))
                n_frames += 1
        reason = info["end_reason"] or ("video_time_limit" if not done else "unknown")
        banner = f"RESULT: {reason.upper()}  ({total:+.0f} points)"
        last = add_hud(render_frame(env), hud, banner=banner)
        for _ in range(int(hold_s * fps)):
            wr.add(last)
            n_frames += 1
        meta["episodes"].append({"title": title, "seed": int(seed), "end_reason": reason, "steps": t,
                                 "points": round(float(total), 2), "people_located": info["people_located"],
                                 "n_people": info["n_people"], "fires_out": info["fires_out"],
                                 "n_fires": info["n_fires"], "collisions": info["collisions"]})
    out = wr.close()
    meta.update({"path": out, "encoder": wr.encoder, "frames": n_frames,
                 "duration_s": round(n_frames / fps, 1), "size_bytes": os.path.getsize(out)})
    return meta
