from pymodbus import __version__ as pymodbus_version
from pymodbus.datastore import (
    ModbusSequentialDataBlock,
    ModbusServerContext,
    ModbusSlaveContext,
    ModbusSparseDataBlock,
)
from pymodbus.device import ModbusDeviceIdentification
from pymodbus.server import StartAsyncSerialServer
from typing import Callable, Literal
from dataclasses import dataclass
from enum import Enum, auto
from loguru import logger
import sys

if sys.version_info >= (3, 12):
    from typing import override
else:

    def override(f):
        return f


SLAVE_ADDR = 0x67

OFFSET = 0x11
# RO
OBJECT_EXISTS_REG = OFFSET + 0x0
# RW
SASH_STATE_REG = OFFSET + 0x1
# RW, default 0
LED_CTRL_REG = OFFSET + 0x2


class SashState(Enum):
    """
    The state of the sash (barrier screen)
    """

    STOP = 0
    RISING = 1
    FALLING = 2


class ObjectExists(Enum):
    NO_OBJECT = 0
    OBJECT_EXISTS = 1


# https://apmonitor.com/dde/index.php/Main/ModbusTransfer
# https://github.com/pymodbus-dev/pymodbus/blob/dev/examples/server_callback.py
# https://github.com/pymodbus-dev/pymodbus/blob/dev/examples/server_async.py
# http://www.simplymodbus.ca/exceptions.htm
# https://product-help.schneider-electric.com/ED/ES_Power/NT-NW_Modbus_IEC_Guide/EDMS/DOCA0054EN/DOCA0054xx/Master_NS_Modbus_Protocol/Master_NS_Modbus_Protocol-5.htm


# we're only use holding registers
class CallbackDataBlock(ModbusSequentialDataBlock):
    """
    A datablock that stores the new value in memory,.
    and passes the operation to a message queue for further processing.
    """

    on_set_sash_state: Callable[[SashState], None] = lambda _: None
    on_set_led_ctrl: Callable[[int], None] = lambda _: None
    _object_exist_lock: bool = True

    def __init__(self):
        """Initialize."""
        super().__init__(0, [0] * (OFFSET + 3))

    def set_object_exists(self, value: ObjectExists):
        """Set the OBJECT_EXISTS_REG."""
        val = 0xFFFF if value == ObjectExists.OBJECT_EXISTS else 0x0000
        self._object_exist_lock = False
        self.setValues(OBJECT_EXISTS_REG, val)
        self._object_exist_lock = True

    @override
    def setValues(self, address: int, values: list[int] | int):
        """Set the requested values of the datastore."""

        def to_int(value: int | list[int]) -> int:
            if isinstance(value, list):
                return int.from_bytes(bytes(value), "big")
            return value

        logger.debug("0x{:04X} <- {}", address, values)
        if address == SASH_STATE_REG:
            self.on_set_sash_state(SashState(to_int(values)))
        elif address == LED_CTRL_REG:
            self.on_set_led_ctrl(to_int(values))
        elif address == OBJECT_EXISTS_REG:
            if self._object_exist_lock:
                logger.error("Cannot set OBJECT_EXISTS_REG")
                return
        super().setValues(address, values)

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
    port: str = "/dev/ttyUSB0"
    baudrate: int = 115_200


# | Primary tables    | Object type | Type of    | Comments                                                      |
# | ----------------- | ----------- | ---------- | ------------------------------------------------------------- |
# | Discretes Input   | Single bit  | Read-Only  | This type of data can be provided by an I/O system.           |
# | Coils             | Single bit  | Read-Write | This type of data can be alterable by an application program. |
# | Input Registers   | 16-bit word | Read-Only  | This type of data can be provided by an I/O system            |
# | Holding Registers | 16-bit word | Read-Write | This type of data can be alterable by an application program. |


ModbusDataBlock = ModbusSequentialDataBlock | ModbusSparseDataBlock


def create_modbus_context(
    di: ModbusDataBlock,
    co: ModbusDataBlock,
    hr: ModbusDataBlock,
    ir: ModbusDataBlock,
    zero_mode: bool = False,
):
    """
    Create a modbus context
    """
    return ModbusSlaveContext(di=di, co=co, hr=hr, ir=ir, zero_mode=zero_mode)  # type: ignore


async def modbus_server_loop(args: RunServerArgs, hr: CallbackDataBlock):
    """Run server setup."""
    # discrete inputs
    # coils initializer
    # holding register
    # input registers
    context = create_modbus_context(
        di=ModbusSequentialDataBlock.create(),
        co=ModbusSequentialDataBlock.create(),
        hr=hr,
        ir=ModbusSequentialDataBlock.create(),
    )
    # slave!
    # https://github.com/pymodbus-dev/pymodbus/issues/561
    slaves = {
        SLAVE_ADDR: context,
    }
    context = ModbusServerContext(slaves=slaves, single=False)
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
