from dataclasses import dataclass
from typing import Generator, Tuple, TypedDict, Optional
from pathlib import Path
from json import JSONEncoder, JSONDecoder
from os import PathLike

import cv2
from cv2.typing import MatLike
import cv2 as cv
import numpy as np
import json
from loguru import logger
from numpy.typing import ArrayLike, DTypeLike, NDArray
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm


class DetectionFeatures(TypedDict):
    x: int
    y: int
    w: int
    h: int
    area: float
    cX: int
    cY: int


@dataclass
class CapProps:
    width: int
    height: int
    channels: int
    fps: float
    frame_count: Optional[int] = None


class CapPropsDict(TypedDict):
    width: int
    height: int
    channels: int
    fps: float
    frame_count: Optional[int]


def fourcc(*args: str) -> int:
    return cv2.VideoWriter_fourcc(*args)  # type: ignore


def video_cap(
    src: PathLike | int | str,
    scale: float = 1,
) -> Tuple[Generator[MatLike, None, None], CapProps]:
    assert 0 < scale <= 1, "scale should be in (0, 1]"
    if isinstance(src, PathLike):
        cap = cv2.VideoCapture(str(src))
    else:
        cap = cv2.VideoCapture(src)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * scale)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * scale)
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    channels = int(cap.get(cv2.CAP_PROP_CHANNEL))
    props = CapProps(width=width,
                     height=height,
                     fps=fps,
                     channels=channels,
                     frame_count=frame_count)

    def gen():
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if scale != 1:
                frame = cv2.resize(frame, (width, height))
            yield frame
        cap.release()

    return gen(), props  # type: ignore


class GaussianStateDict(TypedDict):
    x: list[float]
    P: list[list[float]]


class TrackingDict(TypedDict):
    id: int
    state: GaussianStateDict
    survived_time_steps: int
    missed_time_steps: int


class ResultDict(TypedDict):
    props: CapPropsDict
    detections_history: list[list[DetectionFeatures]]
    confirmed_histories: list[list[TrackingDict]]


def main():
    with open("result.json", "r") as f:
        data:ResultDict = json.load(f)
    frames, props = video_cap("PETS09-S2L1-raw.mp4", scale=0.5)
    cc = fourcc("m", "p", "4", "v")
    out = cv2.VideoWriter("output.mp4", cc, props.fps,
                          (props.width, props.height))

    i = 0
    colors = np.random.randint(0, 255, size=(1024, 3))
    try:
        for frame in tqdm(frames, total=props.frame_count):
            dets = data["detections_history"][i]
            for det in dets:
                x, y, w, h = det["x"], det["y"], det["w"], det["h"]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            trackings = data["confirmed_histories"][i]
            for track in trackings:
                id = track["id"]
                x, y, vx, vy = track["state"]["x"]
                x = int(x)
                y = int(y)
                color_ = colors[id]
                color = tuple(color_.tolist())
                cv.circle(frame, (int(x), int(y)), 5, color, -1)
            out.write(frame)
            i += 1
    except Exception as e:
        logger.exception(e)
    finally:
        out.release()


if __name__ == "__main__":
    main()
