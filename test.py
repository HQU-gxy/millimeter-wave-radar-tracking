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
                            signed=False) & 0x7fff
        x_sign = (data[0] & 0x80) >> 7
        x = x_ if x_sign == 1 else -x_

        y_ = int.from_bytes(data[2:4], byteorder='little',
                            signed=False) & 0x7fff
        y_sign = (data[2] & 0x80) >> 7
        y = y_ if y_sign == 1 else -y_

        speed_sign = (data[4] & 0x80) >> 7
        speed_mag = int.from_bytes(data[4:6], byteorder='little',
                                   signed=False) & 0x7f
        speed = speed_mag if speed_sign == 1 else -speed_mag
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
    target_1: Int[NDArray, "... 2"] = np.empty((0, 2))
    target_2: Int[NDArray, "... 2"] = np.empty((0, 2))
    target_3: Int[NDArray, "... 2"] = np.empty((0, 2))
    target_window = st.empty()
    for targets in app_state["gen"]:
        for i, target in enumerate(targets.targets):
            if i == 0:
                target_1 = np.vstack((target_1, np.array([target.coord])))
            elif i == 1:
                target_2 = np.vstack((target_2, np.array([target.coord])))
            elif i == 2:
                target_3 = np.vstack((target_3, np.array([target.coord])))
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
        fig = go.Figure(data)
        fig.update_layout(showlegend=True)
        target_window.plotly_chart(fig)


if __name__ == '__main__':
    main(port="/dev/cu.usbserial-5")
