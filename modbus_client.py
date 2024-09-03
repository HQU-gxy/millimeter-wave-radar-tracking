from pymodbus.client import AsyncModbusSerialClient
import click
from loguru import logger
import anyio


SLAVE_ADDR = 0x67
OFFSET = 0x11
# RO
OBJECT_EXISTS_REG_R = OFFSET + 0x0
OBJECT_EXISTS_REG_W = OBJECT_EXISTS_REG_R - 1
# RW
SASH_STATE_REG_R = OFFSET + 0x1
SASH_STATE_REG_W = SASH_STATE_REG_R - 1
# RW, default 0
LED_CTRL_REG_R = OFFSET + 0x2
LED_CTRL_REG_W = LED_CTRL_REG_R - 1

async def async_main(port:str):
    client = AsyncModbusSerialClient(port=port, baudrate=115200)
    await client.connect()
    logger.info("Connected")
    toggle = True
    await client.write_register(LED_CTRL_REG_W, int(toggle), slave=SLAVE_ADDR)
    while True:
        await anyio.sleep(1)
        val = await client.read_holding_registers(OBJECT_EXISTS_REG_R, slave=SLAVE_ADDR)
        logger.info("0x{:04X} -> {}", OBJECT_EXISTS_REG_R, val.registers)
        toggle = not toggle
        await client.write_register(LED_CTRL_REG_W, int(toggle), slave=SLAVE_ADDR)
    client.close()

@click.command()
@click.argument("port", type=str)
def main(port: str):
    anyio.run(async_main, port)
    

if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter
