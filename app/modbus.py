from pymodbus import __version__ as pymodbus_version
from pymodbus.datastore import (
    ModbusSequentialDataBlock,
    ModbusServerContext,
    ModbusSlaveContext,
    ModbusSparseDataBlock,
)
from pymodbus.device import ModbusDeviceIdentification
from pymodbus.server import StartAsyncSerialServer
from typing import Callable, Literal, Optional, Tuple, TypedDict, cast, overload
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
    MAX = 2


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
    1   has filter
    0   Object exists
    LSB

    MSB
    7 6  54 32  1  0
    1 0 |AR|SS| F |OE
    """

    result: ArbiterResult
    sash_state: SashState
    has_filtered: bool
    last_valid_result: Optional[ArbiterResult] = None

    @property
    def is_object_exists(self) -> ObjectExists:
        def arbiter_result():
            if self.result == ArbiterResult.INDECISIVE:
                if self.last_valid_result is None:
                    raise ValueError("no last_valid_result when result is indecisive")
                else:
                    if self.last_valid_result == ArbiterResult.INDECISIVE:
                        raise ValueError(
                            "both result and last_valid_result are indecisive"
                        )
                    else:
                        return self.last_valid_result
            else:
                return self.result

        r = arbiter_result()
        if self.sash_state == SashState.FALLING:
            # ignore still when falling
            if r == ArbiterResult.STILL:
                return ObjectExists.NO_OBJECT
            elif r == ArbiterResult.MOVING:
                return ObjectExists.OBJECT_EXISTS
            elif r == ArbiterResult.IDLE:
                return ObjectExists.NO_OBJECT
            else:
                raise ValueError(f"Invalid result: {self.result}")
        elif self.sash_state == SashState.RISING:
            if r == ArbiterResult.IDLE:
                return ObjectExists.NO_OBJECT
            else:
                return ObjectExists.OBJECT_EXISTS
        else:
            if r == ArbiterResult.MOVING or r == ArbiterResult.STILL:
                return ObjectExists.OBJECT_EXISTS
            elif r == ArbiterResult.IDLE:
                return ObjectExists.NO_OBJECT
            else:
                raise ValueError(f"Invalid result: {self.result}")

    def marshal(self) -> int:
        result = 0b1000_0000  # Set bit 7 to 1 and bit 6 to 0
        result |= (self.result.value & 0b11) << 4  # Arbiter result in bits 5-4
        result |= (self.sash_state.value & 0b11) << 2  # Sash state in bits 3-2
        result |= int(self.has_filtered) << 1  # Has filter in bit 1
        result |= self.is_object_exists.value  # Object exists in bit 0
        return result

    @staticmethod
    def unmarshal(data: int) -> Tuple["ObjectExistsDecider", ObjectExists]:
        r = ArbiterResult((data >> 4) & 0b11)
        s = SashState((data >> 2) & 0b11)
        f = bool((data >> 1) & 0b1)
        o = ObjectExists(data & 0b1)
        return ObjectExistsDecider(result=r, sash_state=s, has_filtered=f), o


# https://apmonitor.com/dde/index.php/Main/ModbusTransfer
# https://github.com/pymodbus-dev/pymodbus/blob/dev/examples/server_callback.py
# https://github.com/pymodbus-dev/pymodbus/blob/dev/examples/server_async.py
# http://www.simplymodbus.ca/exceptions.htm
# https://product-help.schneider-electric.com/ED/ES_Power/NT-NW_Modbus_IEC_Guide/EDMS/DOCA0054EN/DOCA0054xx/Master_NS_Modbus_Protocol/Master_NS_Modbus_Protocol-5.htm


@dataclass
class ModbusRegisterCallback:
    read: Callable[[], int]
    write: Callable[[int], None]

    @staticmethod
    def default() -> "ModbusRegisterCallback":
        """
        Create a default callback that always returns 0 and does nothing on write.
        """
        return ModbusRegisterCallback(read=lambda: 0, write=lambda _: None)


RegistersCallbacks = dict[str, ModbusRegisterCallback]


class RadarRegistersCallbacks(TypedDict):
    OE: ModbusRegisterCallback
    SS: ModbusRegisterCallback
    LED: ModbusRegisterCallback


class RadarRegisters:
    REGISTER_NAMES = ["OE", "SS", "LED"]
    CALLBACK_MAP: RegistersCallbacks = {}

    def set_callbacks(self, cb: RadarRegistersCallbacks):
        self.CALLBACK_MAP = cast(RegistersCallbacks, cb)

    def get_by_key(self, key: str) -> int:
        return self.CALLBACK_MAP[key].read()

    def get_by_index(self, index: int) -> int:
        return self.get_by_key(self.REGISTER_NAMES[index])

    def get_by_range(self, start: int, count: int = 1) -> list[int]:
        return [self.get_by_index(i) for i in range(start, start + count)]

    def set_by_key(self, key: str, value: int):
        self.CALLBACK_MAP[key].write(value)

    def set_by_index(self, index: int, value: int):
        self.set_by_key(self.REGISTER_NAMES[index], value)

    def set_by_range(self, start: int, values: list[int]):
        for i, value in enumerate(values):
            self.set_by_index(start + i, value)

    def __len__(self):
        return len(self.REGISTER_NAMES)

    @overload
    def __getitem__(self, key: str) -> int: ...
    @overload
    def __getitem__(self, key: int) -> int: ...
    @overload
    def __getitem__(self, key: slice) -> list[int]: ...

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.get_by_key(key)
        if isinstance(key, int):
            return self.get_by_index(key)
        if isinstance(key, slice):
            return self.get_by_range(key.start, key.stop)
        raise ValueError(f"Invalid key type: {type(key)}")


# we're only use holding registers
class CallbackDataBlock(ModbusSequentialDataBlock):
    """
    A datablock that stores the new value in memory,.
    and passes the operation to a message queue for further processing.
    """

    on_set_sash_state: Callable[[SashState], None] = lambda _: None
    on_set_led_ctrl: Callable[[int], None] = lambda _: None

    _callbacks: RadarRegisters = RadarRegisters()
    _last_valid_arbiter_result: ArbiterResult = ArbiterResult.IDLE
    arbiter_result: ArbiterResult = ArbiterResult.IDLE
    sash_state: SashState = SashState.STOP
    io_state: bool = False
    has_filtered: bool = False

    def __init__(self):
        """Initialize."""
        super().__init__(0, [0] * (OFFSET + 3))

        def oe_w(_: int):
            pass

        def oe_r() -> int:
            d = ObjectExistsDecider(
                result=self.arbiter_result,
                sash_state=self.sash_state,
                has_filtered=self.has_filtered,
                last_valid_result=self._last_valid_arbiter_result,
            )
            logger.info(
                "ar={}; ss={}; oe={}", d.result, d.sash_state, d.is_object_exists
            )
            return d.marshal()

        def ss_w(value: int):
            if value > SashState.MAX.value:
                logger.error("Invalid sash state: {}", value)
                return
            self.sash_state = SashState(value)
            self.on_set_sash_state(self.sash_state)

        def ss_r() -> int:
            return self.sash_state.value

        def led_w(value: int):
            self.io_state = bool(value)
            self.on_set_led_ctrl(self.io_state)

        def led_r() -> int:
            return int(self.io_state)

        self._callbacks.set_callbacks(
            {
                "OE": ModbusRegisterCallback(read=oe_r, write=oe_w),
                "SS": ModbusRegisterCallback(read=ss_r, write=ss_w),
                "LED": ModbusRegisterCallback(read=led_r, write=led_w),
            }
        )

    def set_object_exists(self, value: ArbiterResult):
        """Set the OBJECT_EXISTS_REG."""
        self.arbiter_result = value
        if value != ArbiterResult.INDECISIVE:
            self._last_valid_arbiter_result = value

    @override
    def setValues(self, address: int, values: list[int]):
        """Set the requested values of the datastore."""

        logger.debug("0x{:04X} <- {}", address, values)
        if address < OFFSET:
            raise ValueError(f"Invalid address: {address}")
        self._callbacks.set_by_range(address - OFFSET, values)

    @override
    def getValues(self, address: int, count: int = 1):
        """Return the requested values from the datastore."""
        if address < OFFSET:
            raise ValueError(f"Invalid address: {address}")
        val = self._callbacks.get_by_range(address - OFFSET, count)
        # logger.debug("0x{:04X} ({}) -> {}", address, count, val)
        return val

    @override
    def validate(self, address: int, count: int = 1):
        """Check to see if the request is in range."""
        if address < OFFSET:
            return False
        if count > len(self._callbacks):
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
