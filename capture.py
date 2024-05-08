import cv2
import cv2 as cv
from cv2.typing import MatLike
from typing import Optional, Generator
from dataclasses import dataclass
from datetime import datetime, timedelta
from loguru import logger

@dataclass
class Props:
    width: int
    height: int
    fps: float


VideoGenerator = Generator[MatLike, None, None]


def fourcc(*args: str) -> int:
    return cv2.VideoWriter_fourcc(*args)  # type: ignore


def frame_gen(index: int,
              props: Optional[Props]) -> tuple[VideoGenerator, Props]:
    cap = cv2.VideoCapture(index)
    if props is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, props.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, props.height)
        cap.set(cv2.CAP_PROP_FPS, props.fps)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    props = Props(width=width, height=height, fps=fps)

    def gen():
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield frame

    return gen(), props


def main():
    props_ = Props(width=640, height=480, fps=30)
    frames, props = frame_gen(0, props_)
    logger.info(f"Width: {props.width}, Height: {props.height}, FPS: {props.fps}")
    date = datetime.now()
    output_name = f"video-{date.strftime('%Y-%m-%d-%H-%M-%S')}.mp4"
    logger.info(f"Output: {output_name}")
    writer = cv2.VideoWriter(output_name, fourcc(*"mp4v"), props.fps,
                             (props.width, props.height))
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()


if __name__ == "__main__":
    main()
