import sys
from collections import deque
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Iterable, Optional, Sequence, Literal, override

import anyio
from dataclasses import dataclass
import click
import numpy as np
from anyio import create_memory_object_stream, create_task_group
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from anyio import to_thread, from_thread, open_file, AsyncFile
from loguru import logger
from pydantic import BaseModel
from serial import Serial

from app.gpio import GPIO
from app.stillness_fis import FisInput, infer as infer_stillness
from app.range_fis import infer as infer_range
from capture.model import END_MAGIC, Target, Targets


from pymodbus import __version__ as pymodbus_version
from pymodbus.datastore import (
    ModbusSequentialDataBlock,
    ModbusServerContext,
    ModbusSlaveContext,
    ModbusSparseDataBlock,
)
from pymodbus.device import ModbusDeviceIdentification
from pymodbus.server import (
    StartAsyncSerialServer,
)

StoreType = Literal["sequential", "sparse", "factory"]

# https://apmonitor.com/dde/index.php/Main/ModbusTransfer
# https://github.com/pymodbus-dev/pymodbus/blob/dev/examples/server_callback.py
# https://github.com/pymodbus-dev/pymodbus/blob/dev/examples/server_async.py
# http://www.simplymodbus.ca/exceptions.htm
# https://product-help.schneider-electric.com/ED/ES_Power/NT-NW_Modbus_IEC_Guide/EDMS/DOCA0054EN/DOCA0054xx/Master_NS_Modbus_Protocol/Master_NS_Modbus_Protocol-5.htm


class CallbackDataBlock(ModbusSequentialDataBlock):
    """A datablock that stores the new value in memory,.

    and passes the operation to a message queue for further processing.
    """

    def __init__(self, addr: int, values: list[int]):
        """Initialize."""
        super().__init__(addr, values)

    @override
    def setValues(self, address: int, value: list[int] | int):
        """Set the requested values of the datastore."""
        super().setValues(address, value)

    @override
    def getValues(self, address: int, count: int = 1):
        """Return the requested values from the datastore."""
        result: list[int] = super().getValues(address, count=count)
        return result

    @override
    def validate(self, address: int, count: int = 1):
        """Check to see if the request is in range."""
        result = super().validate(address, count=count)
        return result


@dataclass
class RunServerArgs:
    store_type: StoreType = "sequential"
    port: str = "/dev/ttyUSB0"
    baudrate: int = 9600


# | Primary tables    | Object type | Type of    | Comments                                                      |
# | ----------------- | ----------- | ---------- | ------------------------------------------------------------- |
# | Discretes Input   | Single bit  | Read-Only  | This type of data can be provided by an I/O system.           |
# | Coils             | Single bit  | Read-Write | This type of data can be alterable by an application program. |
# | Input Registers   | 16-bit word | Read-Only  | This type of data can be provided by an I/O system            |
# | Holding Registers | 16-bit word | Read-Write | This type of data can be alterable by an application program. |


async def run_server(args: RunServerArgs):
    """Run server setup."""

    def get_store(store_type: StoreType):
        if store_type == "sequential":
            return ModbusSequentialDataBlock(0, [17] * 100)
        elif store_type == "sparse":
            return ModbusSparseDataBlock({0: 0, 5: 1})
        elif store_type == "factory":
            return ModbusSequentialDataBlock.create()
        else:
            raise ValueError(f"Unknown store type: {store_type}")

    store = get_store(args.store_type)
    # discrete inputs
    # coils initializer
    # holding register
    # input registers
    context = ModbusSlaveContext(di=store, co=store, hr=store, ir=store)  # type: ignore
    # slave!
    # https://github.com/pymodbus-dev/pymodbus/issues/561
    context = ModbusServerContext(slaves=context, single=True)
    identity = ModbusDeviceIdentification(
        info_name={
            "VendorName": "WeiHua",
            "ProductCode": "WH",
            "VendorUrl": "https://weihua-iot.cn",
            "ProductName": "Radar",
            "ModelName": "H3",
            "MajorMinorRevision": pymodbus_version,
        }
    )
    # use RTU framer
    await StartAsyncSerialServer(
        context, identity=identity, port=args.port, baudrate=args.baudrate
    )
