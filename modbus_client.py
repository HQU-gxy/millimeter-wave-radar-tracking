from pymodbus.client import AsyncModbusSerialClient
import click
from loguru import logger
import anyio


SLAVE_ADDR = 0x67
# RO
OBJECT_EXISTS_REG = 0x0
# RW
SASH_STATE_REG = 0x1
# RW, default 0
LED_CTRL_REG = 0x2

async def async_main(port:str):
    client = AsyncModbusSerialClient(port=port, baudrate=115200)
    await client.connect()
    logger.info("Connected")
    await client.write_register(0x1, 0, slave=SLAVE_ADDR)
    logger.info("LED ON")
    client.close()

@click.command()
@click.argument("port", type=str)
def main(port: str):
    anyio.run(async_main, port)
    

if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter
