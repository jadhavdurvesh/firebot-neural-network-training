"""Template for running the brain on the real robot (Jetson / Raspberry Pi / x86 mini-PC).

Fill in the two classes with your drivers (ROS 2 topics, serial to the motor controller, CAN, GPIO ...).
Nothing else changes: memory, observation, ONNX policy and safety layer are the same code used in sim.

    python -m firebot_brain.hardware_template --onnx firebot_policy.onnx --people 2 --fires 1

BEFORE FIRST POWER-ON: wheels off the ground, e-stop within reach, speed cap lowered in RobotConfig.
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from firebot_sim.config import RobotConfig
from .brain import FireBotBrain
from .hal import Command, SensorFrame


class MyRobotSensors:
    def __init__(self, cfg: RobotConfig):
        self.cfg = cfg
        # TODO: open lidar, wheel odometry/IMU (pose), battery monitor, temp sensors, vision/mapping streams

    def read(self) -> SensorFrame:
        # TODO: fill every field from real sensors. Units/frames: see hal.py docstring.
        raise NotImplementedError("implement MyRobotSensors.read() for your hardware")


class MyRobotActuators:
    def send(self, cmd: Command) -> None:
        # TODO: speed setpoint = cmd.throttle * max_speed (m/s); steering = cmd.steer * max_steer (rad); pump = cmd.pump > 0
        raise NotImplementedError

    def stop(self) -> None:
        # TODO: zero speed, centre steering, pump off, brakes on
        raise NotImplementedError


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--people", type=int, default=1)
    ap.add_argument("--fires", type=int, default=1)
    ap.add_argument("--hz", type=float, default=20.0)
    ap.add_argument("--robot-config", default=None)
    a = ap.parse_args()
    cfg = RobotConfig.from_json(a.robot_config) if a.robot_config else RobotConfig()
    brain = FireBotBrain(a.onnx, cfg, a.people, a.fires)
    sensors, act = MyRobotSensors(cfg), MyRobotActuators()
    period = 1.0 / a.hz
    try:
        while True:
            t0 = time.monotonic()
            frame = sensors.read()
            cmd, info = brain.step(frame, now=time.monotonic())
            act.send(cmd)
            if info["safety"]:
                print("safety:", info["safety"])
            time.sleep(max(0.0, period - (time.monotonic() - t0)))
    except KeyboardInterrupt:
        pass
    finally:
        act.stop()                      # always leave the robot stopped


if __name__ == "__main__":
    main()
