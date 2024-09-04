from pymodbus import __version__ as pymodbus_version
from pymodbus.datastore import (
    ModbusSequentialDataBlock,
    ModbusServerContext,
    ModbusSlaveContext,
    ModbusSparseDataBlock,
)
from pymodbus.device import ModbusDeviceIdentification
from pymodbus.server import StartAsyncSerialServer
from typing import Callable, Literal, Optional, Tuple
from dataclasses import dataclass
from enum import Enum, auto
from loguru import logger
from .state import ArbiterResult
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
    """
    The state of the object
    """

    NO_OBJECT = 0
    OBJECT_EXISTS = 1


@dataclass
class ObjectExistsDecider:
    """
    MSB
    7   always 1
    6   always 0
    5-4 Arbiter result
    3-2 Sash state
    1   always 0
    0   Object exists
    LSB
    """

    result: ArbiterResult
    sash_state: SashState
    last_valid_result: Optional[ArbiterResult] = None

    @property
    def is_object_exists(self) -> ObjectExists:
        if self.result == ArbiterResult.MOVING:
            return ObjectExists.OBJECT_EXISTS
        elif self.result == ArbiterResult.STILL:
            return (
                ObjectExists.OBJECT_EXISTS
                if self.sash_state == SashState.STOP
                else ObjectExists.NO_OBJECT
            )
        elif self.result == ArbiterResult.IDLE:
            return ObjectExists.NO_OBJECT
        else:
            if self.last_valid_result is not None:
                return (
                    ObjectExists.NO_OBJECT
                    if self.last_valid_result == ArbiterResult.IDLE
                    else ObjectExists.OBJECT_EXISTS
                )
            else:
                raise ValueError("last_valid_result is None")

    def marshal(self) -> int:
        result = 0b10000000  # Set bit 7 to 1 and bit 6 to 0
        result |= (self.result.value & 0b11) << 4  # Arbiter result in bits 5-4
        result |= (self.sash_state.value & 0b11) << 2  # Sash state in bits 3-2
        result |= self.is_object_exists.value  # Object exists in bit 0
        return result

    @staticmethod
    def unmarshal(data: int) -> Tuple["ObjectExistsDecider", ObjectExists]:
        r = ArbiterResult((data >> 4) & 0b11)
        s = SashState((data >> 2) & 0b11)
        o = ObjectExists(data & 0b1)
        return ObjectExistsDecider(result=r, sash_state=s), o


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

    _last_valid_arbiter_result: ArbiterResult = ArbiterResult.IDLE
    arbiter_result: ArbiterResult = ArbiterResult.IDLE
    sash_state: SashState = SashState.STOP
    io_state: bool = False

    def __init__(self):
        """Initialize."""
        super().__init__(0, [0] * (OFFSET + 3))

    def set_object_exists(self, value: ArbiterResult):
        """Set the OBJECT_EXISTS_REG."""
        self.arbiter_result = value
        if value != ArbiterResult.INDECISIVE:
            self._last_valid_arbiter_result = value

    @override
    def setValues(self, address: int, values: list[int] | int):
        """Set the requested values of the datastore."""

        def to_int(value: int | list[int]) -> int:
            if isinstance(value, list):
                return int.from_bytes(bytes(value), "big")
            return value

        logger.debug("0x{:04X} <- {}", address, values)
        if address == SASH_STATE_REG:
            self.sash_state = SashState(to_int(values))
            self.on_set_sash_state(self.sash_state)
        elif address == LED_CTRL_REG:
            self.io_state = bool(to_int(values))
            self.on_set_led_ctrl(self.io_state)

    @override
    def getValues(self, address: int, count: int = 1):
        """Return the requested values from the datastore."""
        if address == OBJECT_EXISTS_REG:
            payload = ObjectExistsDecider(
                result=self.arbiter_result,
                sash_state=self.sash_state,
                last_valid_result=self._last_valid_arbiter_result,
            )
            return [payload.marshal()]
        elif address == SASH_STATE_REG:
            return [self.sash_state.value]
        elif address == LED_CTRL_REG:
            return [int(self.io_state)]
        else:
            return super().getValues(address, count=count)

    @override
    def validate(self, address: int, count: int = 1):
        """Check to see if the request is in range."""
        if address < OFFSET:
            return False
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
