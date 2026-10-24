import hashlib

from firebot_sim.config import RobotConfig
from firebot_sim.robot import build_urdf, urdf_path


def test_urdf_path_rebuilds_empty_cached_file(tmp_path, monkeypatch):
    monkeypatch.setattr("firebot_sim.robot.tempfile.gettempdir", lambda: str(tmp_path))
    cfg = RobotConfig()
    xml = build_urdf(cfg)
    h = hashlib.md5(xml.encode()).hexdigest()[:10]
    cached = tmp_path / f"firebot_{h}.urdf"
    cached.write_text("", encoding="utf-8")

    path = urdf_path(cfg)

    assert path == str(cached)
    assert cached.stat().st_size > 0
    assert '<robot name="firebot">' in cached.read_text(encoding="utf-8")
