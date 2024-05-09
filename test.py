import datetime
import json
import serial
import click
from loguru import logger
from dataclasses import dataclass
from pydantic import BaseModel, Field, PrivateAttr
from anyio import create_task_group
from anyio.to_thread import run_sync
from typing import Annotated, Final, Literal, Tuple, List, Optional, TypedDict, Generator, AsyncGenerator
from collections import deque
import numpy as np
import plotly.graph_objects as go
from plotly.graph_objects import Scatter
from jaxtyping import Num, Float, Int
from datetime import datetime, timedelta
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
import jsonlines
import colorsys
import streamlit as st

NDArray = np.ndarray


class Target(BaseModel, frozen=True):
    """
    in millimeters, uint_16_t (MSB is the sign bit) in little endian
    """
    coord: Tuple[int, int]
    """
    only magnitude, in cm/s, uint16_t (MSB is the sign bit)
    """
    speed: int
    # uint16_t
    resolution: Annotated[int, Field(exclude=True)]

    @property
    def coord_si(self) -> Tuple[float, float]:
        return self.coord[0] / 1000, self.coord[1] / 1000

    @property
    def speed_si(self) -> float:
        return self.speed / 100

    @staticmethod
    def unmarshal(data: bytes) -> Optional["Target"]:
        assert len(data) == 8, "Invalid data length"
        if data == bytes([0, 0, 0, 0, 0, 0, 0, 0]):
            return None

        def list_hex(data: bytes):
            return " ".join(f"{b:02x}" for b in data)

        def msb_bit_int16(num: int) -> int:
            """
            Some genius decided to use the most significant bit as the sign bit

            Parameters:
                num (int): A 16-bit number (unsigned)
            Returns:
                int: A 16-bit number (use most significant bit as sign bit)
            """
            assert 0 <= num < 2**16
            sign = num & 0x8000 >= 1
            n = num & 0x7fff
            return n if sign else -n

        x_ = int.from_bytes(data[0:2], byteorder='little', signed=False)
        # since it's little endian, the most significant bit is in the last byte (data[1])
        x = msb_bit_int16(x_)

        y_ = int.from_bytes(data[2:4], byteorder='little', signed=False)
        y = msb_bit_int16(y_)

        speed_ = int.from_bytes(data[4:6], byteorder='little', signed=False)

        speed = msb_bit_int16(speed_)
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
    assert targets.targets[0]._resolution == 320
    print(targets)


class Params(BaseModel, frozen=True):
    port: str
    baudrate: int


class AppState(TypedDict):
    ser: serial.Serial
    gen: Generator[Targets, None, None]


@st.cache_resource
def resource(_params: Params) -> AppState:
    now = datetime.now()
    ser = serial.Serial(_params.port, _params.baudrate)
    logger.info(
        f"Serial port {_params.port} opened with baudrate {_params.baudrate}")
    SAMPLE_INTERVAL_MS = 88
    SAMPLE_INTERVAL = timedelta(milliseconds=SAMPLE_INTERVAL_MS)
    file_name = f"radar_{now.strftime('%Y%m%d_%H%M%S')}_{SAMPLE_INTERVAL_MS}ms.jsonl"
    logger.info(f"Saving data to {file_name}")
    writer = jsonlines.open(file_name, mode="w", flush=True)

    def gen():
        while True:
            data = ser.read_until(bytes([0x55, 0xcc]))
            try:
                targets = Targets.unmarshal(data)
            except ValueError as e:
                logger.exception(e)
                continue
            if targets is not None:
                writer.write(targets.model_dump())
                yield targets

    return {"ser": ser, "gen": gen()}


def generate_colors(time_list: list[datetime],
                    start_rgb: Int[NDArray, "3"],
                    end_rgb: Int[NDArray, "3"],
                    interpolation: Literal["hsv", "rgb"] = "rgb"):
    try:
        time_normalized = [(dt - min(time_list)).total_seconds() /
                           (max(time_list) - min(time_list)).total_seconds()
                           for dt in time_list]
    except ZeroDivisionError:
        time_normalized = [0.5] * len(
            time_list)  # Default to mid color if division by zero

    if interpolation == "hsv":
        # Convert RGB to HSV
        start_hsv = colorsys.rgb_to_hsv(start_rgb[0], start_rgb[1],
                                        start_rgb[2])
        end_hsv = colorsys.rgb_to_hsv(end_rgb[0], end_rgb[1], end_rgb[2])

        # Interpolate in HSV space
        colors_hsv = [
            (
                start_hsv[0] +
                (end_hsv[0] - start_hsv[0]) * t,  # Hue interpolation
                start_hsv[1],  # Saturation kept constant
                start_hsv[2]  # Value kept constant
            ) for t in time_normalized
        ]

        # Convert HSV back to RGB
        colors_rgb = [colorsys.hsv_to_rgb(h, s, v) for h, s, v in colors_hsv]

        # Convert RGB to hex
        colors_hex = [
            "#%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255))
            for r, g, b in colors_rgb
        ]
        return colors_hex
    elif interpolation == "rgb":
        colors = [start_rgb * (1 - t) + end_rgb * t for t in time_normalized
                 ]  # Linear interpolation between colors
        colors_hex = [
            "#%02x%02x%02x" %
            (int(c[0] * 255), int(c[1] * 255), int(c[2] * 255)) for c in colors
        ]
        return colors_hex
    else:
        raise ValueError("Invalid interpolation method")


def main(port: str, baudrate: int = 256000):
    params = Params(port=port, baudrate=baudrate)
    app_state = resource(params)
    st.title("Radar Target Tracking")
    WINDOW_SIZE = 100
    targets: deque[tuple[Targets, datetime]] = deque(maxlen=WINDOW_SIZE)

    target_window = st.empty()
    speed_window = st.empty()
    # Define RGB for color schemes
    blue_to_red = np.array([[0, 0, 1], [1, 0, 0]])  # Blue to Red
    green_to_pink = np.array([[0, 1, 0], [1, 0, 1]])  # Green to Pink
    purple_to_orange = np.array([[128 / 255, 0, 128 / 255],
                                 [1, 165 / 255, 0]])  # Purple to Orange
    for tgs in app_state["gen"]:
        last = datetime.now()
        targets.append((tgs, datetime.now()))
        tg1 = [t.targets[0].coord_si for t, _ in targets if len(t.targets) > 0]
        tg1_v = [
            t.targets[0].speed_si for t, _ in targets if len(t.targets) > 0
        ]
        tg1_time = [dt for t, dt in targets if len(t.targets) > 0]
        tg1_colors_hex = generate_colors(tg1_time, blue_to_red[0],
                                         blue_to_red[1])
        tg2 = [t.targets[1].coord_si for t, _ in targets if len(t.targets) > 1]
        tg2_v = [
            t.targets[1].speed_si for t, _ in targets if len(t.targets) > 1
        ]
        tg2_time = [dt for t, dt in targets if len(t.targets) > 1]
        tg2_colors_hex = generate_colors(tg2_time, green_to_pink[0],
                                         green_to_pink[1])
        tg3 = [t.targets[2].coord_si for t, _ in targets if len(t.targets) > 2]
        tg3_v = [
            t.targets[2].speed_si for t, _ in targets if len(t.targets) > 2
        ]
        tg3_time = [dt for t, dt in targets if len(t.targets) > 2]
        tg3_colors_hex = generate_colors(tg3_time, purple_to_orange[0],
                                         purple_to_orange[1])
        data = {
            "data": [
                Scatter(x=list(map(lambda x: x[0], tg1)),
                        y=list(map(lambda x: x[1], tg1)),
                        mode="markers",
                        name="Target 1",
                        marker=dict(color=tg1_colors_hex)),
                Scatter(x=list(map(lambda x: x[0], tg2)),
                        y=list(map(lambda x: x[1], tg2)),
                        marker=dict(color=tg2_colors_hex),
                        mode="markers",
                        name="Target 2"),
                Scatter(x=list(map(lambda x: x[0], tg3)),
                        y=list(map(lambda x: x[1], tg3)),
                        marker=dict(color=tg3_colors_hex),
                        mode="markers",
                        name="Target 3"),
            ],
        }
        data_vel = {
            "data": [
                Scatter(x=tg1_time, y=tg1_v, mode="lines", name="Target 1"),
                Scatter(x=tg2_time, y=tg2_v, mode="lines", name="Target 2"),
                Scatter(x=tg3_time, y=tg3_v, mode="lines", name="Target 3"),
            ],
        }

        fig = go.Figure(data)
        fig.update_layout(showlegend=True)
        fig.update_xaxes(range=[-3, 3])
        fig.update_yaxes(range=[0, 8])
        fig.update_xaxes(title_text="X (m)")
        fig.update_yaxes(title_text="Y (m)")

        fig_vel = go.Figure(data_vel)
        fig_vel.update_layout(showlegend=True)
        fig_vel.update_xaxes(title_text="Sample number")
        fig_vel.update_yaxes(title_text="Speed (m/s)")
        fig_vel.update_yaxes(range=[-1, 1])

        target_window.plotly_chart(fig)
        speed_window.plotly_chart(fig_vel)


if __name__ == '__main__':
    main(port="/dev/cu.usbserial-0001")
    # test_unmarshal()
