from pymodbus.client import AsyncModbusSerialClient
import click
from loguru import logger
from app.modbus import ObjectExistsDecider
import anyio


SLAVE_ADDR = 0x67
OFFSET = 0x10
# RO
OBJECT_EXISTS_REG = OFFSET + 0x0
# RW
SASH_STATE_REG = OFFSET + 0x1
# RW, default 0
LED_CTRL_REG = OFFSET + 0x2


async def async_main(port: str):
    client = AsyncModbusSerialClient(port=port, baudrate=115200)
    await client.connect()
    logger.info("Connected")
    toggle = True
    await client.write_register(LED_CTRL_REG, int(toggle), slave=SLAVE_ADDR)
    while True:
        await anyio.sleep(1)
        val = await client.read_holding_registers(OBJECT_EXISTS_REG, slave=SLAVE_ADDR)
        value = val.registers[0]
        logger.info("0x{:04X} -> {}", OBJECT_EXISTS_REG, value)
        dec, o = ObjectExistsDecider.unmarshal(value)
        logger.info("ObjectExistsDecider={}; E={}", dec, o)
        toggle = not toggle
        await client.write_register(LED_CTRL_REG, int(toggle), slave=SLAVE_ADDR)
    client.close()


@click.command()
@click.argument("port", type=str)
def main(port: str):
    anyio.run(async_main, port)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
