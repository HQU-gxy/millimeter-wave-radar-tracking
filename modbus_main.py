from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Literal, Optional, Sequence, Tuple
from app import modbus
from app.state import ArbiterResult, DoorSignal, DoorState, MaybeTarget
from app.modbus import (
    modbus_server_loop,
    CallbackDataBlock,
    RunServerArgs,
    SashState,
    ObjectExists,
    ObjectExistsDecider,
)

import anyio
import click
import numpy as np
from anyio import (
    AsyncFile,
    create_memory_object_stream,
    create_task_group,
    open_file,
    to_thread,
)
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from loguru import logger
from pydantic import BaseModel
from serial import Serial

from app.gpio import GPIO
from app.range_fis import infer as infer_range
from app.stillness_fis import FisInput
from app.stillness_fis import infer as infer_stillness
from capture.model import END_MAGIC, Target, Targets

is_debug_radar = False

MAX_SIZE = 5
ORIGIN_POINT = (0, 0)
holding_registers = CallbackDataBlock()
io = GPIO()


def on_led_ctrl(value: int):
    if value > 0:
        logger.info("LED set")
        io.high()
    else:
        logger.info("LED unset")
        io.low()


holding_registers.on_set_led_ctrl = on_led_ctrl

sash_state: SashState = SashState.STOP


def on_screen_movement_set(value: SashState):
    global sash_state
    logger.info("sash state set to {}", value)
    sash_state = value


holding_registers.on_set_sash_state = on_screen_movement_set


async def gen_target(serial: Serial):
    TIMEOUT = 0.2
    serial.timeout = TIMEOUT
    while True:
        try:
            with anyio.fail_after(TIMEOUT):
                data = await to_thread.run_sync(serial.read_until, END_MAGIC)
                if data:
                    targets = Targets.unmarshal(data)
                    yield targets
        except TimeoutError:
            logger.warning("serial read timeout")
        except ValueError as e:
            logger.error("serial read error: {}", e)
            continue


def check_anyio_version():
    """
    Check if the anyio version is greater than 4.3
    Throws an AssertionError if the version is less than 4.3
    """
    from importlib.metadata import version

    anyio_version_str = version("anyio").split(".")
    anyio_version = tuple([int(x) for x in anyio_version_str])
    assert (
        anyio_version[0] == 4 and anyio_version[1] >= 3
    ), "anyio version must be >= 4.3"


async def infer_loop(
    ser: Serial,
    tx: MemoryObjectSendStream[ArbiterResult],
    writer: Optional[AsyncFile] = None,
):
    logger.info("infer block started")
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

    def calc_distance(p1: Sequence[int], p2: Sequence[int]):
        assert len(p1) == 2, "p1 must be a 2-tuple"
        assert len(p2) == 2, "p2 must be a 2-tuple"
        return float(np.linalg.norm(np.array(p1) - np.array(p2)))

    with tx:
        async for targets in gen_target(ser):

            def cond(t: Target) -> bool:
                res = infer_range(t.coord[0], t.coord[1])
                if is_debug_radar:
                    logger.debug("infer_range({}, {})={}", t.coord[0], t.coord[1], res)
                return res > -0.5

            tgs = targets.targets
            before_len = len(tgs)
            filtered_tgs = list(filter(cond, targets.targets))
            after_len = len(tgs)
            if before_len != after_len:
                logger.warning("before={}; after={}", targets, Targets(targets=tgs))

            # find the distance that is closest to the origin point
            if len(filtered_tgs) == 0:
                logger.warning("no target detected")
                t = None
            elif len(filtered_tgs) == 1:
                t = targets.targets[0]
                if is_debug_radar:
                    logger.info("single target {}", t)
            else:
                t = min(
                    filtered_tgs, key=lambda x: calc_distance(x.coord, ORIGIN_POINT)
                )
                if is_debug_radar:
                    logger.warning("multiple targets {}; selected {}", targets, t)
            t_ = MaybeTarget(target=t, timestamp=datetime.now())
            queue.append(t_)
            # jsonl
            if writer is not None:
                await writer.write(t_.model_dump_json() + "\n")
            # note that deque in python will automatically remove the oldest element
            # which is quite a strange behavior compared to other languages
            # but it avoids the need to pop the oldest element manually
            if len(queue) < MAX_SIZE:
                await tx.send(ArbiterResult.INDECISIVE)
                continue
            if all_none(queue):
                await tx.send(ArbiterResult.IDLE)
            elif all_available(queue):
                x_avg = float(
                    np.mean([t.target.coord[0] for t in queue if t.target is not None])
                )
                y_avg = float(
                    np.mean([t.target.coord[1] for t in queue if t.target is not None])
                )
                speed_avg_abs = np.abs(
                    [t.target.speed for t in queue if t.target is not None]
                )
                speed_avg = float(np.mean(speed_avg_abs))
                speed_std = float(np.std(speed_avg_abs))
                fis_in = FisInput(
                    xAvg=x_avg, yAvg=y_avg, speedMean=speed_avg, speedStd=speed_std
                )
                result = infer_stillness(fis_in)
                if is_debug_radar:
                    logger.info(f"Input={fis_in}; Result={result}")
                if result > 0:
                    await tx.send(ArbiterResult.STILL)
                else:
                    await tx.send(ArbiterResult.MOVING)
            else:
                await tx.send(ArbiterResult.INDECISIVE)


async def action_loop(
    queue: MemoryObjectReceiveStream[ArbiterResult],
):
    logger.info("action loop started")
    async with queue:
        async for result in queue:
            if result != ArbiterResult.INDECISIVE:
                holding_registers.set_object_exists(result)


def find_serial_port() -> Tuple[Optional[str], Optional[str]]:
    from serial.tools.list_ports import comports

    radar_port = None
    modbus_port = None
    for port, _, _ in comports():
        if "radar" in port.lower() or "ch340" in port.lower():
            radar_port = port
        if "modbus" in port.lower() or "cp210" in port.lower():
            modbus_port = port
    return radar_port, modbus_port


def print_ports():
    from serial.tools.list_ports import comports

    for port, desc, hwid in comports():
        print(f"{port}: {desc} ({hwid})")


@click.command()
@click.option("--port", type=str, default="", help="Serial port for Radar")
@click.option("--modbus-port", type=str, default="", help="Serial port for Modbus")
@click.option("--baudrate", type=int, default=256_000, help="Baudrate")
@click.option("--modbus-baudrate", type=int, default=115_200, help="Modbus Baudrate")
@click.option("-o", "--output", type=str, help="Output file", default=None)
@click.option("--overwrite", is_flag=True, help="Overwrite the output file")
@click.option("--debug-radar", is_flag=True, help="Debug radar")
def main(
    port: str,
    modbus_port: str,
    baudrate: int,
    modbus_baudrate: int,
    output: Optional[str] = None,
    overwrite: bool = False,
    debug_radar: bool = False,
):
    global is_debug_radar
    is_debug_radar = debug_radar
    if port == "" or modbus_port == "":
        logger.info("either radar port or modbus port is not specified; try to find it")
        rp, mp = find_serial_port()
        if rp is None or mp is None:
            logger.error("Cannot find the serial port")
            print_ports()
            exit(1)
        port = rp
        modbus_port = mp
    logger.info("Radar port={}; Modbus port={}", port, modbus_port)
    check_anyio_version()
    if output is None:
        logger.info("no output file specified, not output the result to file")
        output_path = None
    else:
        output_path = Path(output)
        if output_path.exists() and not overwrite:
            logger.error(
                f"Output file {output} exists, use --overwrite to overwrite the file"
            )
            return
        if output_path.is_dir():
            logger.error(f"{output} is a directory")
            return
        logger.info(f"Output file: {output}")

    async def _block(result_tx: MemoryObjectSendStream[ArbiterResult]):
        with Serial(port, baudrate) as ser:
            if output_path is not None:
                async with await open_file(output_path, "w", encoding="utf-8") as f:
                    await infer_loop(ser, result_tx, f)
            else:
                await infer_loop(ser, result_tx)

    async def _main():
        # https://github.com/agronholm/anyio/discussions/521
        async with create_task_group() as tg:
            io.low()
            result_tx, result_rx = create_memory_object_stream[ArbiterResult](0)
            tg.start_soon(action_loop, result_rx)
            modbus_args = RunServerArgs(port=modbus_port, baudrate=modbus_baudrate)
            tg.start_soon(modbus_server_loop, modbus_args, holding_registers)
            await _block(result_tx)

    anyio.run(_main)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
