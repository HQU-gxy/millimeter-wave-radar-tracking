import serial
import click
from loguru import logger
from dataclasses import dataclass
from pydantic import BaseModel
from anyio import create_task_group
from anyio.to_thread import run_sync
from typing import Final, Tuple, List, Optional, TypedDict, Generator, AsyncGenerator
import numpy as np
import plotly.graph_objects as go
from plotly.graph_objects import Scatter
from jaxtyping import Num, Float, Int
import streamlit as st

NDArray = np.ndarray


class Target(BaseModel, frozen=True):
    # seems little endian
    coord: Tuple[int, int]  # in millimeters, uint_16_t (MSB is the sign bit)
    speed: int  # only magnitude, in cm/s, uint16_t (MSB is the sign bit)
    resolution: int  # uint16_t

    @staticmethod
    def unmarshal(data: bytes) -> Optional["Target"]:
        assert len(data) == 8, "Invalid data length"
        if data == bytes([0, 0, 0, 0, 0, 0, 0, 0]):
            return None
        x_ = int.from_bytes(data[0:2], byteorder='little',
                            signed=False)
        # since it's little endian, the most significant bit is in the last byte (data[1])
        x_sign = x_ & 0x8000 >= 1
        x__ = x_ & 0x7fff
        x = x__ if x_sign else -x__

        y_ = int.from_bytes(data[2:4], byteorder='little',
                            signed=False)
        y_sign = y_ & 0x8000 >= 1
        y__ = y_ & 0x7fff
        y = y__ if y_sign else -y__

        speed_ = int.from_bytes(data[4:6], byteorder='little',
                                signed=False)

        def list_hex(data: bytes):
            return " ".join(f"{b:02x}" for b in data)

        speed_sign = speed_ & 0x8000 >= 1
        speed__ = speed_ & 0x7fff
        speed = speed__ if speed_sign else -speed__
        resolution = int.from_bytes(data[6:8], byteorder="little", signed=False)
        return Target(coord=(x, y), speed=speed, resolution=resolution)


class Targets(BaseModel, frozen=True):
    MAGIC: Final[bytes] = bytes([0xaa, 0xff, 0x03, 0x00])
    targets: List[Target] = []

    @staticmethod
    def unmarshal(data: bytes):
        offset = 0
        if data[0:4] != Targets.MAGIC:
            raise ValueError("Invalid magic")
        offset += 4
        targets = []
        # we have three targets
        # if the data is all zeros, it means the target is not set
        for _ in range(3):
            target = Target.unmarshal(data[offset:offset + 8])
            if target is not None:
                targets.append(target)
            offset += 8
        return Targets(targets=targets)


def test_unmarshal():
    data = bytes([
        0xaa, 0xff, 0x03, 0x00, 0x0e, 0x03, 0xb1, 0x86, 0x10, 0x00, 0x40, 0x01,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00
    ])
    targets = Targets.unmarshal(data)
    assert len(targets.targets) == 1
    assert targets.targets[0].coord == (-782, 1713)
    assert targets.targets[0].speed == -16
    assert targets.targets[0].resolution == 320
    print(targets)


class Params(BaseModel, frozen=True):
    port: str
    baudrate: int


class AppState(TypedDict):
    ser: serial.Serial
    gen: Generator[Targets, None, None]


@st.cache_resource
def resource(_params: Params) -> AppState:
    ser = serial.Serial(_params.port, _params.baudrate)
    logger.info(
        f"Serial port {_params.port} opened with baudrate {_params.baudrate}")

    def gen():
        while True:
            data = ser.read_until(bytes([0x55, 0xcc]))
            targets = Targets.unmarshal(data)
            if targets is not None:
                yield targets

    return {"ser": ser, "gen": gen()}


def main(port: str, baudrate: int = 256000):
    params = Params(port=port, baudrate=baudrate)
    app_state = resource(params)
    st.title("Radar Target Tracking")
    target_1: Int[NDArray, "... 2"] = np.empty((0, 2), dtype=np.float32)
    target_2: Int[NDArray, "... 2"] = np.empty((0, 2), dtype=np.float32)
    target_3: Int[NDArray, "... 2"] = np.empty((0, 2), dtype=np.float32)
    target_1_vel = np.empty((1,), dtype=np.float32)
    target_2_vel = np.empty((1,), dtype=np.float32)
    target_3_vel = np.empty((1,), dtype=np.float32)

    resolution = np.empty((1,))

    target_window = st.empty()
    speed_window = st.empty()
    res_window = st.empty()
    for targets in app_state["gen"]:
        COORD_MAX = 10
        SPEED_MAX = 1_00
        for i, target in enumerate(targets.targets):
            MM_2_M = 1 / 1_000
            CM_2_M = 1 / 1_000
            if i == 0:
                target_1 = np.vstack(
                    (target_1, np.array([target.coord]) * MM_2_M))[-COORD_MAX:]
                target_1_vel = np.append(target_1_vel,
                                         target.speed * CM_2_M)[-SPEED_MAX:]
                resolution = np.append(resolution,
                                       target.resolution)[-SPEED_MAX:]
            elif i == 1:
                target_2 = np.vstack(
                    (target_2, np.array([target.coord]) * MM_2_M))[-COORD_MAX:]
                target_2_vel = np.append(target_2_vel,
                                         target.speed * CM_2_M)[-SPEED_MAX:]
            elif i == 2:
                target_3 = np.vstack(
                    (target_3, np.array([target.coord]) * MM_2_M))[-COORD_MAX:]
                target_3_vel = np.append(target_3_vel,
                                         target.speed * CM_2_M)[-SPEED_MAX:]
        data = {
            "data": [
                Scatter(x=target_1[:, 0],
                        y=target_1[:, 1],
                        mode="markers",
                        name="Target 1"),
                Scatter(x=target_2[:, 0],
                        y=target_2[:, 1],
                        mode="markers",
                        name="Target 2"),
                Scatter(x=target_3[:, 0],
                        y=target_3[:, 1],
                        mode="markers",
                        name="Target 3"),
            ],
        }
        data_vel = {
            "data": [
                Scatter(x=np.arange(len(target_1_vel)),
                        y=target_1_vel,
                        mode="lines",
                        name="Target 1"),
                Scatter(x=np.arange(len(target_2_vel)),
                        y=target_2_vel,
                        mode="lines",
                        name="Target 2"),
                Scatter(x=np.arange(len(target_3_vel)),
                        y=target_3_vel,
                        mode="lines",
                        name="Target 3"),
            ],
        }
        data_res = {
            "data": [
                Scatter(x=np.arange(len(resolution)),
                        y=resolution,
                        mode="lines",
                        name="Resolution"),
            ],
        }

        fig = go.Figure(data)
        fig.update_layout(showlegend=True)
        fig.update_xaxes(range=[-3, 3])
        fig.update_yaxes(range=[-3, 3])
        fig.update_xaxes(title_text="X (m)")
        fig.update_yaxes(title_text="Y (m)")

        fig_vel = go.Figure(data_vel)
        fig_vel.update_layout(showlegend=True)
        fig_vel.update_xaxes(title_text="Sample number")
        fig_vel.update_yaxes(title_text="Speed (m/s)")
        fig_vel.update_yaxes(range=[-1, 1])

        target_window.plotly_chart(fig)
        speed_window.plotly_chart(fig_vel)
        res_window.plotly_chart(go.Figure(data_res))


if __name__ == '__main__':
    main(port="/dev/cu.usbserial-0001")
    # test_unmarshal()
