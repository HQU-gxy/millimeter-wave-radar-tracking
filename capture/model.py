from typing import (
    Annotated,
    AsyncGenerator,
    Final,
    Generator,
    List,
    Literal,
    Optional,
    Tuple,
    TypedDict,
)

from pydantic import BaseModel, Field, PrivateAttr

END_MAGIC: Final = bytes([0x55, 0xCC])


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
            n = num & 0x7FFF
            return n if sign else -n

        x_ = int.from_bytes(data[0:2], byteorder="little", signed=False)
        # since it's little endian, the most significant bit is in the last byte (data[1])
        x = msb_bit_int16(x_)

        y_ = int.from_bytes(data[2:4], byteorder="little", signed=False)
        y = msb_bit_int16(y_)

        speed_ = int.from_bytes(data[4:6], byteorder="little", signed=False)

        speed = msb_bit_int16(speed_)
        resolution = int.from_bytes(data[6:8], byteorder="little", signed=False)
        return Target(coord=(x, y), speed=speed, resolution=resolution)


class Targets(BaseModel, frozen=True):
    MAGIC: Final[bytes] = bytes([0xAA, 0xFF, 0x03, 0x00])
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
