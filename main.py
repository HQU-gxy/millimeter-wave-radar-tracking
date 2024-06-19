from asyncio import to_thread
from typing import Iterable, Optional
from capture.model import Target, Targets, END_MAGIC
from app.fis import infer, FisInput
from app.gpio import GPIO
from serial import Serial
from loguru import logger
from collections import deque
from pydantic import BaseModel
from datetime import datetime, timedelta
import click
import numpy as np
import anyio
from enum import Enum, auto
from anyio.to_thread import run_sync as thread_run_sync
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from anyio import create_task_group, create_memory_object_stream

# low: no detection
# high: detection
#
# GPIO from low to high, door up immediately
# GPIO from high to low, door down after 50s


class ArbiterResult(Enum):
    """
    The result of the arbiter
    """
    INDECISIVE = auto()
    """
    reserved
    """
    MOVING = auto()
    """
    object present and moving actively
    """
    STILL = auto()
    """
    object present but not moving actively
    """
    IDLE = auto()
    """
    no object present
    """


class DoorState(Enum):
    OPEN = auto()
    CLOSED = auto()


class DoorSignal(Enum):
    UP = auto()
    DOWN = auto()


class MaybeTarget(BaseModel, frozen=True):
    target: Optional[Target]
    timestamp: datetime


def gen_target(serial: Serial):
    while True:
        data = serial.read_until(END_MAGIC)
        if data:
            targets = Targets.unmarshal(data)
            yield targets


MAX_SIZE = 16


def check_anyio_version():
    """
    Check if the anyio version is greater than 4.3
    Throws an AssertionError if the version is less than 4.3
    """
    from importlib.metadata import version
    anyio_version_str = version("anyio").split(".")
    anyio_version = tuple([int(x) for x in anyio_version_str])
    assert anyio_version[0] == 4 and anyio_version[
        1] >= 3, "anyio version must be >= 4.3"


def infer_block(ser: Serial, tx: MemoryObjectSendStream[ArbiterResult]):
    queue = deque[MaybeTarget](maxlen=MAX_SIZE)

    # if all of the targets are None, then the object is not present for sure
    # we can send the IDLE signal
    #
    # if the queue is not full or not all of the targets are None/Available
    # we're give indecisive signal
    #
    # we're only care about target 0 for now
    def all_none(targets: Iterable[MaybeTarget]) -> bool:
        return all(t.target is None for t in targets)

    def all_available(targets: Iterable[MaybeTarget]) -> bool:
        return all(t.target is not None for t in targets)

    def send_silent(result: ArbiterResult):
        try:
            tx.send_nowait(result)
        except anyio.WouldBlock:
            ...

    for targets in gen_target(ser):
        try:
            t = targets.targets[0]
        except IndexError:
            t = None
        logger.info(f"Target: {t}")
        queue.append(MaybeTarget(target=t, timestamp=datetime.now()))
        # note that deque in python will automatically remove the oldest element
        # which is quite a strange behavior compared to other languages
        # but it avoids the need to pop the oldest element manually
        if len(queue) < MAX_SIZE:
            send_silent(ArbiterResult.INDECISIVE)
            continue
        if all_none(queue):
            send_silent(ArbiterResult.IDLE)
        elif all_available(queue):
            x_avg = float(
                np.mean(
                    [t.target.coord[0] for t in queue if t.target is not None]))
            y_avg = float(
                np.mean(
                    [t.target.coord[1] for t in queue if t.target is not None]))
            speed_avg = float(
                np.mean([t.target.speed for t in queue if t.target is not None
                        ]))
            speed_std = float(
                np.std([t.target.speed for t in queue if t.target is not None]))
            input = FisInput(xAvg=x_avg,
                             yAvg=y_avg,
                             speedMean=speed_avg,
                             speedStd=speed_std)
            logger.info(f"Input: {input}")
            result = infer(input)
            if result > 0:
                send_silent(ArbiterResult.STILL)
            else:
                send_silent(ArbiterResult.MOVING)
        else:
            send_silent(ArbiterResult.INDECISIVE)


async def action_loop(door: MemoryObjectSendStream[DoorSignal],
                      queue: MemoryObjectReceiveStream[ArbiterResult]):
    async for result in queue:
        if result == ArbiterResult.MOVING:
            logger.info("Object is moving")
            await door.send(DoorSignal.UP)
        elif result == ArbiterResult.STILL:
            logger.info("Object is still")
        elif result == ArbiterResult.IDLE:
            logger.info("No object present")
            await door.send(DoorSignal.DOWN)
        elif result == ArbiterResult.INDECISIVE:
            logger.info("Indecisive")


async def door_loop(door: MemoryObjectReceiveStream[DoorSignal]):
    state = DoorState.CLOSED
    io = GPIO()
    io.low()
    async for signal in door:
        if signal == DoorSignal.UP:
            if state == DoorState.CLOSED:
                io.high()
                state = DoorState.OPEN
        elif signal == DoorSignal.DOWN:
            if state == DoorState.OPEN:
                io.low()
                state = DoorState.CLOSED


@click.command()
@click.argument("port", type=str, help="Serial port", default="/dev/ttyUSB0")
@click.option("--baudrate", type=int, default=256_000, help="Baudrate")
def main(port: str, baudrate: int):
    check_anyio_version()
    with Serial(port, baudrate) as ser:
        result_tx, result_rx = create_memory_object_stream[ArbiterResult]()
        door_tx, door_rx = create_memory_object_stream[DoorSignal]()
        anyio.run(action_loop, door_tx, result_rx)
        anyio.run(door_loop, door_rx)
        # sync looping, will block the main thread
        try:
            infer_block(ser, result_tx)
        except KeyboardInterrupt:
            ...


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
